import argparse
import os
import json
import ast
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers.cache_utils import DynamicCache

from train_kv_prefix import KVPrefixModel


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


def generate_from_embedding(model, tokenizer, embedding_tensor, max_new_tokens, do_sample, temperature, top_p):
    model.eval()
    with torch.no_grad():
        past_key_values = model.projector(embedding_tensor)
        pkv = []
        for k, v in past_key_values:
            pkv.append((k.to(model.decoder.dtype), v.to(model.decoder.dtype)))
        past_key_values = DynamicCache.from_legacy_cache(tuple(pkv))

        bsz = embedding_tensor.size(0)
        start_pos = getattr(model, "num_kv_tokens", pkv[0][0].shape[2])
        input_ids = torch.full(
            (bsz, 1),
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
            device=embedding_tensor.device,
            dtype=torch.long,
        )
        attention_mask = torch.ones((bsz, 1), dtype=torch.long, device=embedding_tensor.device)
        cache_position = torch.arange(start_pos, start_pos + input_ids.shape[1], device=embedding_tensor.device)

        outputs = model.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = KVPrefixModel(
        decoder_name=args.decoder_name,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_kv_tokens=args.num_kv_tokens,
        model_dtype=model_dtype,
        use_lora=False,
    ).to(device)

    if args.lora_adapter_dir and os.path.isdir(args.lora_adapter_dir):
        model.decoder = PeftModel.from_pretrained(model.decoder, args.lora_adapter_dir).to(device)

    load_projector_weights(model, args.projector_path, device)

    df = pd.read_csv(args.test_csv)
    num_samples = min(args.num_samples, len(df))

    os.makedirs(os.path.dirname(args.output_log) or ".", exist_ok=True)
    with open(args.output_log, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            row = df.iloc[i]
            initial_text = row["dialogue"]
            embedding = _parse_embedding_to_np(row["embedding"]) 

            emb_tensor = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)

            generated_text = generate_from_embedding(
                model,
                tokenizer,
                emb_tensor,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            entry = f"Initial text:\n{initial_text}\n\nGenerated text:\n{generated_text}\n\n\n"
            f.write(entry)
            print(entry, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="embds_data/test.csv")
    parser.add_argument("--decoder_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--projector_path", type=str, default="outputs_kv_prefix/kv_projector_final.pt")
    parser.add_argument("--output_log", type=str, default="infer_kv_prefix.txt")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_kv_tokens", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_adapter_dir", type=str, default="outputs_kv_prefix/lora_adapter", help="Path to trained LoRA adapter directory")
    args = parser.parse_args()
    main(args)


