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
    arr = np.fromstring(s, sep=" ", dtype=np.float32)
    return arr


class EmbeddingDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dialogue = row["dialogue"]
        embedding = _parse_embedding_to_np(row["embedding"])

        tokens = self.tokenizer(
            dialogue,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        ids = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        seq_len = ids.size(0)
        L = int(mask.sum().item())
        if L == 0:
            ids[0] = eos_id
            mask[0] = 1
        elif L < seq_len:
            ids[L] = eos_id
            mask[L] = 1
        else:
            ids[-1] = eos_id
            mask[-1] = 1

        return {
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "input_ids": ids,
            "attention_mask": mask,
        }


def collate_batch(batch):
    embeddings = torch.stack([item["embedding"] for item in batch])
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]

    lengths = [int(am.sum().item()) for am in attention_mask_list]
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        max_len = 1

    batch_size = len(batch)
    pad_id = 0

    padded_input_ids = input_ids_list[0].new_full((batch_size, max_len), pad_id)
    padded_attention_mask = attention_mask_list[0].new_zeros((batch_size, max_len))

    for i, (ids, L) in enumerate(zip(input_ids_list, lengths)):
        L = int(L)
        padded_input_ids[i, :L] = ids[:L]
        padded_attention_mask[i, :L] = 1

    return {
        "embedding": embeddings,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
    }


class MLPProjector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=1536, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * output_dim),
        )

        self.ln = nn.LayerNorm(output_dim)

    def forward(self, embeddings):
        e = torch.nn.functional.normalize(embeddings, dim=-1) * self.alpha
        projected = self.mlp(e).view(-1, self.num_tokens, self.output_dim)
        return self.ln(projected)


class MLPDecoderModel(nn.Module):
    def __init__(
        self,
        decoder_name,
        emb_dim=768,
        hidden_dim=2048,
        num_soft_tokens=8,
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
            for param in self.decoder.parameters():
                param.requires_grad = False

        decoder_dim = self.decoder.config.hidden_size
        self.projector = MLPProjector(emb_dim, hidden_dim, decoder_dim, num_soft_tokens)
        self.num_soft_tokens = num_soft_tokens

    def forward(self, embeddings, input_ids, attention_mask):
        soft_tokens = self.projector(embeddings).to(self.decoder.dtype)

        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([soft_tokens, inputs_embeds], dim=1)

        soft_attention = torch.ones(
            soft_tokens.shape[0],
            self.num_soft_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        combined_attention = torch.cat([soft_attention, attention_mask], dim=1)

        soft_labels = torch.full(
            (input_ids.shape[0], self.num_soft_tokens),
            -100,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        masked_labels = input_ids.masked_fill(attention_mask == 0, -100)
        combined_labels = torch.cat([soft_labels, masked_labels], dim=1)

        B, L = input_ids.size()
        pos = (
            torch.arange(self.num_soft_tokens + L, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        outputs = self.decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            position_ids=pos,
            labels=combined_labels,
        )

        return outputs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        embeddings = batch["embedding"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(embeddings, input_ids, attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = EmbeddingDataset(args.train_csv, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # , collate_fn=collate_batch
    )

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = MLPDecoderModel(
        args.decoder_name,
        args.emb_dim,
        args.hidden_dim,
        args.num_soft_tokens,
        model_dtype=model_dtype,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    params = list(model.projector.parameters())
    if args.use_lora:
        lora_params = [p for p in model.decoder.parameters() if p.requires_grad]
        params += lora_params
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    with open("train_loss.txt", "a") as f:
        f.write(f"lr: {args.lr}, epochs: {args.epochs}\n")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

        with open("train_loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {train_loss:.4f}\n")

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

    torch.save(
        model.projector.state_dict(), f"{args.output_dir}/mlp_projector_final.pt"
    )
    print(f"Training complete. Model saved to {args.output_dir}")

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
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
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
