import time
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def calculate_inference_speed(start_time, end_time, sequence_length):
    inference_time = end_time - start_time
    tokens_per_second = sequence_length / inference_time
    return tokens_per_second

def run_inference(
    model_name: str,
    input_text: str,
    max_length: int,
    use_kv_cache=False, 
    load_in_4bit=False,
    load_in_8bit=False,
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load model in both 4-bit and 8-bit quantization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.max_memory_allocated()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", quantization_config=quantization_config)
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", ).to(device)
    model.eval()
    if use_kv_cache:
        model.config.use_cache = True
    else:
        model.config.use_cache = False
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_kv_cache,
        )
        end_time = time.time()
        end_mem = torch.cuda.max_memory_allocated()
    max_memory_allocated = end_mem - start_mem
    tokens_per_sec = calculate_inference_speed(start_time, end_time, len(output[0]))
    return tokens_per_sec, max_memory_allocated