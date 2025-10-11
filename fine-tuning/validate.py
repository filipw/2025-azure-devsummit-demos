import json
import os
import time
from mlx_lm.utils import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from typing import List, Dict, Tuple
from huggingface_hub.utils.tqdm import disable_progress_bars

# suppress noisy hugging face stuff
disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_validation_data(file_path: str) -> List[Dict[str, str]]:
    examples = []
    print("Loading validation examples...")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.replace('*', '"'))
            text = data["text"]
            user_part = text.split('<|user|>')[1].split('<|end|>')[0]
            assistant_part = text.split('<|assistant|>')[1].split('<|end|>')[0]
            examples.append({
                "input": user_part,
                "expected": assistant_part
            })
    return examples

def run_inference(model, tokenizer, prompt: str, base_model: bool = False) -> Tuple[str, float]:
    start_time = time.time()
    
    if base_model:
        system_prompt = """You control a music player. You can use these functions:
- play_song(title): Play a specific song
- play_list(title): Play a specific playlist
- pause: Pause playback
- stop: Stop playback
- next: Skip to next track
- prev: Go to previous track
- vol_up: Increase volume
- vol_down: Decrease volume
- mute: Mute audio
- unmute: Unmute audio

You should respond with a function call in the format: fn:function_name "parameter" (where needed and ALWAYS in lowercase). Don't say anything else beyond that function call format.
For example: fn:play_song "bohemian rhapsody" or fn:play_list "workout mix" or fn:next. In all other cases you respond with "Sorry I cannot help with that"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    sampler = make_sampler(0.0)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=50)
    
    if '<|assistant|>' in response:
        response = response.split('<|assistant|>')[1]
    
    if '<|end|>' in response:
        response = response.split('<|end|>')[0]
        
    return response.strip(), time.time() - start_time

def evaluate_model(model_path: str, adapter_path: str | None, validation_data: List[Dict[str, str]], model_name: str):
    results = {"perfect": 0, "command": 0, "total": 0, "time": 0.0}
    total_examples = len(validation_data)
    index_width = len(str(total_examples))
    
    print(f"\n=== Testing {model_name} ===")
    model, tokenizer = load(model_path, adapter_path=adapter_path if adapter_path else None)
    
    for i, example in enumerate(validation_data, 1):
        input_text = example["input"]
        expected = example["expected"]
        
        actual, duration = run_inference(
            model, 
            tokenizer,
            input_text, 
            base_model=(adapter_path is None)
        )
        
        results["total"] += 1
        perfect_match = expected.strip() == actual.strip()
        results["perfect"] += int(perfect_match)
        results["time"] += duration
        
        print(f"[{i:{index_width}d}/{total_examples}] {'✅' if perfect_match else '❌'} '{input_text}' → {actual}")
    
    print(f"\nSummary: {results['perfect']}/{total_examples} correct ({results['perfect'] / total_examples:.1%}), avg {results['time'] / total_examples:.1f}s per request")
    
    del model
    del tokenizer
    return results

def main():
    MODEL_PATH = "microsoft/Phi-3.5-mini-instruct"
    ADAPTER_PATH = "adapters"
    VALIDATION_FILE = "data/valid.jsonl"
    
    print("=== Starting Validation ===")
    validation_data = load_validation_data(VALIDATION_FILE)
    
    finetuned_results = evaluate_model(MODEL_PATH, ADAPTER_PATH, validation_data, "Fine-tuned Model")
    print("\n" + "="*50 + "\n")
    base_results = evaluate_model(MODEL_PATH, None, validation_data, "Base Model")

if __name__ == "__main__":
    main()