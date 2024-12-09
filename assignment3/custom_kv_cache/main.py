import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from customized_gpt2 import CustomizedGPT2LMHeadModel

@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)
        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time


@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time

@torch.no_grad()
def customized_greedy_decoding_w_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    past_key_values = None
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model_w_cache(**tokenized_batch, use_cache=True, past_key_values=past_key_values)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        past_key_values = outputs['past_key_values']
        tokenized_batch['input_ids'] = output_tokens
        tokenized_batch['attention_mask'] = torch.ones_like(output_tokens)
        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time

if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 1 #16
    times = [0, 0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", use_cache=False, device_map="cuda")
    custom_model_w_cache = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", use_cache=True, device_map="cuda")

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)
        custom_cache_res, custom_cache_time = customized_greedy_decoding_w_cache(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time
        times[2] += custom_cache_time

        assert torch.equal(golden_wo_cache_res, custom_res), "custom_model decoding results are not equal"
        assert torch.equal(golden_wo_cache_res, custom_cache_res), "custom_model_w_cache decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])
    print("Time taken for customized greedy decoding with KV cache: ", times[2])
