import yaml
import argparse
from inference import run_inference
import torch


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    torch.cuda.empty_cache()
    tokens_per_sec, mem = run_inference(
        model_name=config['model_name'],
        input_text=config['input_text'],
        max_length=config['max_length'],
        use_kv_cache=config['use_kv_cache'],
        load_in_4bit=config['load_in_4bit'],
        load_in_8bit=config['load_in_8bit'],
    )
    print(f"config: {config}")
    print(f"Tokens per second: {tokens_per_sec}")
    print(f"Memory used: {mem}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    args = parser.parse_args()
    main(args.config)
    
    