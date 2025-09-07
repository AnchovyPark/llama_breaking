import os
import sys
import torch
import nvtx
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer

def main():
    print("=" * 50)
    print("Llama 3.2 1B Model Profiling for Nsight Systems")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/Users/anchovy-mac/Desktop/calculating/model_py/model_file"
    
    print(f"Loading model from {model_path}")
    tokenizer_path = os.path.join(model_path, "tokenizer")
    model_path = os.path.join(model_path, "model")
    
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.save_pretrained(tokenizer_path)
    
    if os.path.exists(model_path):
        config = LlamaConfig.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
    else:
        config = LlamaConfig(
            vocab_size=128256,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=131072,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=True,
            rope_theta=500000.0,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False
        )
        model = LlamaForCausalLM(config)
    
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {device}")
    
    print("\n=== Warmup (3 iterations) ===")
    with torch.no_grad():
        for i in range(3):
            with nvtx.annotate(f"Warmup_{i}", color="turquoise"):
                warmup_input = tokenizer("Hello world", return_tensors="pt")
                input_ids = warmup_input.input_ids.to(device)
                
                _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                print(f"  Warmup {i+1}/3 completed")
    
    print("\n=== Main Profiling ===")
    
    num_input_tokens = 1000
    num_output_tokens = 10
    
    print(f"Input: {num_input_tokens} tokens")
    print(f"Generating {num_output_tokens} tokens...")
    
    input_ids = torch.tensor([[100] * num_input_tokens], device=device, dtype=torch.long)
    
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        with nvtx.annotate("Main_Inference", color="aqua"):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=num_output_tokens,
                do_sample=False,
                use_cache=True
            )
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
    
    latency = (end_time - start_time) * 1000
    tokens_per_second = num_output_tokens / (latency / 1000)
    
    print(f"\n=== Results ===")
    print(f"Latency: {latency:.2f} ms")
    print(f"Tokens/sec: {tokens_per_second:.2f}")
    
    generated_output_ids = outputs[0][-num_output_tokens:]
    print(f"\nGenerated output token IDs: {generated_output_ids.tolist()}")
    
    print("\n" + "=" * 50)
    print("Profiling completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()