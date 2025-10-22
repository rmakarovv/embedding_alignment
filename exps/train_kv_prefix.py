import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def _parse_embedding_to_np(value):
    if isinstance(value, (list, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    s = str(value).strip()
    try:
        obj = json.loads(s)
        return np.asarray(obj, dtype=np.float32)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        return np.asarray(obj, dtype=np.float32)
    except Exception:
        pass
    s = s.strip("[]").replace(",", " ")
    return np.fromstring(s, sep=" ", dtype=np.float32)


class EmbeddingToTextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_target_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["dialogue"]
        embedding = _parse_embedding_to_np(row["embedding"])

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].squeeze(0)
        labels_attn = tokenized["attention_mask"].squeeze(0)

        return {
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "labels": labels,
            "labels_attention_mask": labels_attn,
        }


class KVProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_kv_tokens: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_tokens = num_kv_tokens

        out_dim = num_layers * 2 * num_kv_heads * num_kv_tokens * head_dim
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ln = nn.LayerNorm(head_dim)

    def forward(self, embeddings: torch.Tensor) -> tuple:
        bsz = embeddings.size(0)
        e = torch.nn.functional.normalize(embeddings, dim=-1) * self.alpha
        raw = self.mlp(e)
        shaped = raw.view(
            bsz,
            self.num_layers,
            2,
            self.num_kv_heads,
            self.num_kv_tokens,
            self.head_dim,
        )
        shaped = self.ln(shaped)

        past_key_values = []
        for layer in range(self.num_layers):
            k = shaped[:, layer, 0]
            v = shaped[:, layer, 1]
            past_key_values.append((k, v))
        return tuple(past_key_values)


class KVPrefixModel(nn.Module):
    def __init__(
        self,
        decoder_name,
        emb_dim=768,
        hidden_dim=2048,
        num_kv_tokens=16,
        model_dtype=torch.float16,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
    ):
        super().__init__()
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_name, torch_dtype=model_dtype
        )

        if use_lora:
            if lora_target_modules is None or (
                isinstance(lora_target_modules, (list, tuple))
                and len(lora_target_modules) == 0
            ):
                lora_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            if isinstance(lora_target_modules, str):
                lora_target_modules = [
                    m.strip() for m in lora_target_modules.split(",") if m.strip()
                ]
            lconf = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.decoder = get_peft_model(self.decoder, lconf)
        else:
            for p in self.decoder.parameters():
                p.requires_grad = False

        config = self.decoder.config
        num_layers = getattr(
            config, "num_hidden_layers", getattr(config, "n_layer", None)
        )
        num_attn_heads = getattr(
            config, "num_attention_heads", getattr(config, "n_head", None)
        )
        num_kv_heads = getattr(config, "num_key_value_heads", num_attn_heads)
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        if num_layers is None or num_attn_heads is None or hidden_size is None:
            raise ValueError(
                "Unsupported model config; missing heads/layers/hidden_size"
            )
        head_dim = hidden_size // num_attn_heads

        self.projector = KVProjector(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_kv_tokens=num_kv_tokens,
        )
        self.num_kv_tokens = num_kv_tokens
        self.num_layers = num_layers

    def forward(self, embeddings, input_ids, attention_mask):
        past_key_values = self.projector(embeddings)
        pkv_cast = []
        for k, v in past_key_values:
            pkv_cast.append((k.to(self.decoder.dtype), v.to(self.decoder.dtype)))
        past_key_values = DynamicCache.from_legacy_cache(tuple(pkv_cast))

        bsz, L = input_ids.shape
        past_len = self.num_kv_tokens
        pos = (
            torch.arange(past_len, past_len + L, device=input_ids.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        cache_position = torch.arange(past_len, past_len + L, device=input_ids.device)

        masked_labels = input_ids.masked_fill(attention_mask == 0, -100)

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            past_key_values=past_key_values,
            cache_position=cache_position,
            labels=masked_labels,
            use_cache=True,
        )
        return outputs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        embeddings = batch["embedding"].to(device)
        labels = batch["labels"].to(device)
        labels_attn = batch["labels_attention_mask"].to(device)

        outputs = model(embeddings, labels, labels_attn)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = EmbeddingToTextDataset(args.train_csv, tokenizer, args.max_target_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = KVPrefixModel(
        decoder_name=args.decoder_name,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_kv_tokens=args.num_kv_tokens,
        model_dtype=model_dtype,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    params = list(model.projector.parameters())
    if args.use_lora:
        lora_params = [p for p in model.decoder.parameters() if p.requires_grad]
        params += lora_params
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    with open("train_loss.txt", "a") as f:
        f.write(f"[kv_prefix] lr: {args.lr}, epochs: {args.epochs}\n")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")
        with open("train_loss.txt", "a") as f:
            f.write(f"[kv_prefix] Epoch {epoch+1}, Loss: {train_loss:.4f}\n")

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "projector_state_dict": model.projector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pt",
            )

    torch.save(model.projector.state_dict(), f"{args.output_dir}/kv_projector_final.pt")
    print(f"Training complete. Projector saved to {args.output_dir}")

    with open("train_loss.txt", "a") as f:
        f.write("--------------------------------\n")

    if args.use_lora:
        lora_dir = args.lora_adapter_dir or os.path.join(
            args.output_dir, "lora_adapter"
        )
        os.makedirs(lora_dir, exist_ok=True)
        model.decoder.save_pretrained(lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="embds_data/train.csv")
    parser.add_argument(
        "--decoder_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument("--output_dir", type=str, default="outputs_kv_prefix")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_kv_tokens", type=int, default=16)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target module names for LoRA",
    )
    parser.add_argument(
        "--lora_adapter_dir",
        type=str,
        default="",
        help="Path to trained LoRA adapter directory",
    )

    args = parser.parse_args()
    main(args)
