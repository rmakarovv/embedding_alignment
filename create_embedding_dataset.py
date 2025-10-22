import os

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

DATASET_NAME = "d0rj/dialogsum-ru"
MODEL_NAME = "ai-forever/ruBert-base"
COLUMN = "dialogue"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_texts_cls(
    texts, tokenizer, model, device, batch_size: int = 16, max_length: int = 512
):
    embeddings = []
    for start in tqdm.trange(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
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


def main():
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    ds_dict = load_dataset(DATASET_NAME)
    os.makedirs("embeddings/dialogsum-ru", exist_ok=True)

    for split, ds in ds_dict.items():
        if COLUMN not in ds.column_names:
            raise ValueError(
                f"Column '{COLUMN}' not found in split '{split}'. Available: {ds.column_names}"
            )

        texts = ds[COLUMN]
        embeddings = embed_texts_cls(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        ds = ds.add_column("embedding", embeddings)
        split_out_dir = os.path.join("embeddings/dialogsum-ru", split)
        os.makedirs(split_out_dir, exist_ok=True)
        ds.save_to_disk(split_out_dir)
        print(f"Saved split '{split}' with embeddings to {split_out_dir}")


if __name__ == "__main__":
    main()
