# Assignment3
## LLM Inference
### basic comparison
first, run this command: 
```
cd llm_inference
```
control the experiment setup by modifying the llm_inference/config/config.yaml: 
```
model_name: "gpt2"
max_length: 200
input_text: "This is the story of Cinderella:"
use_kv_cache: False
load_in_4bit: False
load_in_8bit: True
```
run the experiment by running the command: 
```
python main.py --config config/config.yaml
```
### custom kv cache
first, run this command:
```
cd custom_kv_cache
```
then run this command:
```
python main.py
```
## LLM Reasoning
first, run this command:
```
cd llm_reasoning
```
running the experiment by passing the following three arguments: 
- datasets: currently only support 'gsm8k'
- technique: support 'naive', 'cot', 'icl' and 'reflexion'
- api_key
```
python main.py --datasets gsm8k --technique reflexion --api_key YOUR_API_KEY
```