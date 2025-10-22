import json
import faiss
from pathlib import Path
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.emb_inv_decoding import EmbInvDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device
import torch

long_passages = False
add_noise = False

class NoisyEncoder(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        self.noise_level = kwargs.get('noise_level', 0)
        super().__init__(*args, **kwargs)

    def encode(self, text, add_special_tokens=True, **kwargs):
        embs = super().encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return embs + torch.randn_like(embs) * self.noise_level
    
def emb_inv_experiment(should_natural=False, encoder_name='gte', beam_width=5, max_steps=20, top_k=50, top_p=1):
    """
    Run an experiment to test retrieval-augmented generation.
    
    Args:
        should_natural (bool): Whether to enforce naturalness in the outputs
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
    """
    print("Running Embedding Inversion experiment...")
    
    # Set up trigger and control text
    trigger = "tesla"
    control_text = "I hate {trig}."
    repetition_penalty = 1.5

    import torch

    # Автоматический выбор устройства
    if torch.cuda.is_available():
        file_device = "cuda"
    elif torch.backends.mps.is_available():
        file_device = "mps"  # GPU на Mac
    else:
        file_device = "cpu"

    print(f"Using device: {file_device}")

    encoder = NoisyEncoder("thenlper/gte-base", device=file_device)

    if encoder_name == 'gte':
        encoder = NoisyEncoder("thenlper/gte-base", device=file_device)
    elif encoder_name == 'gte-Qwen':
        encoder = NoisyEncoder("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device=file_device, trust_remote_code=True)
    elif encoder_name == 'gtr':
        encoder = NoisyEncoder("sentence-transformers/gtr-t5-base", device=file_device)
    elif encoder_name == 'contriever':
        encoder = NoisyEncoder("facebook/contriever", device=file_device)
    elif encoder_name == 'rubert':
        encoder = NoisyEncoder("DeepPavlov/rubert-base-cased-sentence", device=file_device)
    
    # Initialize decoding strategy
    attack = EmbInvDecoding(
        encoder=encoder, 
        device=file_device,
        should_natural=should_natural,
        repetition_penalty=repetition_penalty
    )
    
    # Sample test prompts
    
    from datasets import load_dataset

    if long_passages:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en")
        docs = ds['train'].shuffle(seed=42).select(range(100))
        filtered_docs = []
        for doc in docs:
            doc_len = len(encoder.tokenizer.encode(doc['text']))
            if doc_len >= 512:
                filtered_docs.append(doc['text'])
        target_docs = filtered_docs[:10]
        print("filtered docs")
        print(len(filtered_docs))
    else:
        csv_path = Path(__file__).resolve().parents[2] / "test.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {csv_path}")
        csv_dataset = load_dataset("csv", data_files=str(csv_path))
        csv_dataset = csv_dataset['train'].shuffle(seed=42)
        target_docs = [row['dialogue'] for row in csv_dataset if row.get('dialogue')]
    
    adv_texts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = f'./data/emb_inv_attack_{"natural" if should_natural else "unnatural"}_{encoder_name}_{"long" if long_passages else "short"}_{"no_noise" if not add_noise else "noise"}_{timestamp}.json'

    if not long_passages:
        max_len_arr = [max_steps]
    else:
        max_len_arr = [16, 32, 64, 128, 256, 512]

    if add_noise:
        noise_levels = [0.001, 0.01, 0.1]
    else:
        noise_levels = [0]
    
    # Run decoding for each prompt
    for full_target in target_docs[:5]:
        for max_len in max_len_arr:
            for noise_level in noise_levels:
                attack.encoder.noise_level = noise_level
                print("-" * 50)
                target = encoder.tokenizer.decode(encoder.tokenizer.encode(full_target, add_special_tokens=False)[:max_len])
                print(f"Processing prompt: {target}")
                
                # Set up combined scorer
                
                # Run decoding
                prompt = 'Расскажи историю и отвечай только на русском языке.'
                for idx in range(1):
                    if long_passages:
                        if idx == 0:
                            current_top_k = top_k
                            current_max_len = 32
                        else:
                            current_top_k = 10
                            current_max_len = max_len
                    else:
                        current_top_k = top_k
                        current_max_len = max_len
                    best_cand = attack.run_decoding(
                        prompt=prompt,
                        target=target,
                        beam_width=beam_width,
                        max_steps=current_max_len,
                        top_k=current_top_k,
                        top_p=top_p,
                        should_full_sent=False,
                        verbose=False,
                        randomness=True
                    )
                    print(best_cand.token_ids)
                    print([best_cand.seq_str], 'cos_sim:', best_cand.cos_sim, 'bleu_score:', sentence_bleu([target], best_cand.seq_str))
                    prompt = f"Напиши на русском предложение, похожее на это: {best_cand.seq_str}"
                
                # Save results
                result_str = best_cand.seq_str
                adv_texts.append(result_str)
                print(f"Result: {result_str}")
                attack.llm_wrapper.template_example()
                bleu_score = sentence_bleu([target], result_str)
                
                append_to_target_dir(target_dir, {
                    'target': target,
                    'generation': result_str,
                    'cos_sim': best_cand.cos_sim or 0.0,
                    'naturalness': best_cand.naturalness or 0.0,
                    'bleu_score': bleu_score,
                    'beam_width': beam_width,
                    'max_steps': max_len,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'encoder_name': encoder_name,
                    'noise_level': noise_level
                })
