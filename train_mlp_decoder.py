import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


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
        embedding = np.fromstring(row["embedding"].strip("[]"), sep=" ")

        return {
            "dialogue": dialogue,
            "embedding": embedding.astype(np.float32),
        }


def make_collate_fn(tokenizer, max_length: int = 512):
    def collate(batch):
        dialogues = [b.get("dialogue", "") if isinstance(b.get("dialogue", ""), str) else "" for b in batch]

        embeddings_np = np.stack([b["embedding"] for b in batch], axis=0)
        embeddings = torch.tensor(embeddings_np, dtype=torch.float32)

        enc = tokenizer(
            dialogues,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "embedding": embeddings,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    return collate


class MLPProjector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=1536, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * output_dim),
        )

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        projected = self.mlp(embeddings)
        soft_tokens = projected.view(batch_size, self.num_tokens, self.output_dim)
        return soft_tokens


class MLPDecoderModel(nn.Module):
    def __init__(
        self,
        decoder_name,
        emb_dim=768,
        hidden_dim=2048,
        num_soft_tokens=8,
        model_dtype=torch.float16,
    ):
        super().__init__()
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_name, torch_dtype=model_dtype, attn_implementation="flash_attention_2"
            ).eval()
        except Exception:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_name, torch_dtype=model_dtype
            ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        for param in self.decoder.parameters():
            param.requires_grad = False

        decoder_dim = self.decoder.config.hidden_size
        self.projector = MLPProjector(emb_dim, hidden_dim, decoder_dim, num_soft_tokens)
        self.num_soft_tokens = num_soft_tokens

    def forward(self, embeddings, input_ids, attention_mask, sentence=False):
        soft_tokens = self.projector(embeddings).to(self.decoder.dtype)
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids).detach()
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

        # mask pad tokens in labels so they do not contribute to the loss
        masked_labels = input_ids.masked_fill(attention_mask == 0, -100)
        combined_labels = torch.cat([soft_labels, masked_labels], dim=1)

        if not sentence:
            outputs = self.decoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                labels=combined_labels,
            )
        else:
            gen_ids = self.decoder.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=512,
            )

            outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return outputs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        embeddings = batch["embedding"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(embeddings, input_ids, attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def embed_texts_cls(texts, tokenizer, model, device, max_length: int = 512):
    embeddings = []
    batch_texts = texts
    batch_texts = [t if isinstance(t, str) else "" for t in batch_texts]
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        cls_embds = outputs.last_hidden_state[:, 0, :]
    embeddings.extend(cls_embds.detach().cpu().tolist())
    return embeddings


def eval_epoch(model, dataloader, device, rubert_tokenizer, rubert_model):
    model.eval()
    cosine_sims = []
    batch_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch_idx >= 10:
                break
            batch_idx += 1

            embeddings = batch["embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(embeddings, input_ids, attention_mask, sentence=True)

            bert_embeddings = embed_texts_cls(
                outputs, rubert_tokenizer, rubert_model, device
            )
            cosine_sims.extend(
                cosine_similarity(embeddings.cpu().numpy(), bert_embeddings)
            )

    avg_cosine = float(np.mean(cosine_sims)) if len(cosine_sims) > 0 else 0.0
    return avg_cosine


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = EmbeddingDataset(args.train_csv, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
    )

    eval_dataset = EmbeddingDataset(args.eval_csv, tokenizer, args.max_length)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
    )

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = MLPDecoderModel(
        args.decoder_name,
        args.emb_dim,
        args.hidden_dim,
        args.num_soft_tokens,
        model_dtype=model_dtype,
    )
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=args.lr)

    rubert_tokenizer = AutoTokenizer.from_pretrained(args.rubert_name)
    rubert_model = AutoModel.from_pretrained(args.rubert_name)
    rubert_model = rubert_model.to(device).eval()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

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

        eval_cosine_similarity = eval_epoch(
            model, eval_loader, device, rubert_tokenizer, rubert_model
        )
        print(
            f"Epoch {epoch+1}/{args.epochs}, Cosine Similarity: {eval_cosine_similarity:.4f}"
        )
        with open("eval_cosine_similarity.txt", "a") as f:
            f.write(f"{epoch+1}, {eval_cosine_similarity:.4f}\n")

    torch.save(
        model.projector.state_dict(), f"{args.output_dir}/mlp_projector_final.pt"
    )
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="embds_data/train.csv")
    parser.add_argument("--eval_csv", type=str, default="embds_data/val.csv")
    parser.add_argument(
        "--decoder_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--rubert_name", type=str, default="DeepPavlov/rubert-base-cased"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=2)

    args = parser.parse_args()
    main(args)
