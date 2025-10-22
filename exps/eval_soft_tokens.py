import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from train_soft_tokens import MLPDecoderModel


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


def load_projector_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "projector_state_dict" in state:
        state = state["projector_state_dict"]
    model.projector.load_state_dict(state)


def generate_from_embedding(
    model, tokenizer, embedding_tensor, max_new_tokens, do_sample, temperature, top_p
):
    model.eval()
    with torch.no_grad():
        soft_tokens = model.projector(embedding_tensor).to(model.decoder.dtype)
        attention_prefix = torch.ones(
            soft_tokens.shape[0],
            soft_tokens.shape[1],
            dtype=torch.long,
            device=soft_tokens.device,
        )

        outputs = model.decoder.generate(
            inputs_embeds=soft_tokens,
            attention_mask=attention_prefix,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = MLPDecoderModel(
        decoder_name=args.decoder_name,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_soft_tokens=args.num_soft_tokens,
        model_dtype=model_dtype,
        use_lora=False,
    ).to(device)

    if args.lora_adapter_dir and os.path.isdir(args.lora_adapter_dir):
        model.decoder = PeftModel.from_pretrained(
            model.decoder, args.lora_adapter_dir
        ).to(device)

    load_projector_weights(model, args.projector_path, device)

    df = pd.read_csv(args.test_csv)
    num_samples = min(args.num_samples, len(df))

    with open(args.output_log, "w", encoding="utf-8") as f:
        cosine_similarities = []
        for i in range(num_samples):
            row = df.iloc[i]
            initial_text = row["dialogue"]
            embedding = _parse_embedding_to_np(row["embedding"])

            emb_tensor = torch.tensor(
                embedding, dtype=torch.float32, device=device
            ).unsqueeze(0)

            generated_text = generate_from_embedding(
                model,
                tokenizer,
                emb_tensor,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            rubert_tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
            rubert = AutoModel.from_pretrained("ai-forever/ruBert-base").to(device)
            rubert.eval()

            enc_initial_text = rubert_tokenizer(
                initial_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc_generated_text = rubert_tokenizer(
                generated_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            enc_initial_text = {k: v.to(device) for k, v in enc_initial_text.items()}
            enc_generated_text = {
                k: v.to(device) for k, v in enc_generated_text.items()
            }
            with torch.no_grad():
                out_initial_text = rubert(**enc_initial_text, return_dict=True)
                out_generated_text = rubert(**enc_generated_text, return_dict=True)
                initial_text_embedding = out_initial_text.last_hidden_state[:, 0, :]
                generated_text_embedding = out_generated_text.last_hidden_state[:, 0, :]
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    initial_text_embedding, generated_text_embedding, dim=-1
                ).item()
                cosine_similarities.append(cosine_similarity)
                entry = f"Initial text:\n{initial_text}\n\nGenerated text:\n{generated_text}\n\nCosine similarity: {cosine_similarity}\n\n\n"

            f.write(entry)
            print(entry, end="\n\n\n")

        f.write(f"Average cosine similarity: {np.mean(cosine_similarities)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="embds_data/test.csv")
    parser.add_argument(
        "--decoder_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--projector_path", type=str, default="outputs/mlp_projector_final.pt"
    )
    parser.add_argument("--output_log", type=str, default="infer_prev.txt")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_adapter_dir",
        type=str,
        default="outputs/lora_adapter",
        help="Path to trained LoRA adapter directory",
    )
    args = parser.parse_args()
    main(args)
