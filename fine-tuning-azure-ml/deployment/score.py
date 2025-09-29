import os
import re
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def init():
    """
    Called once when the container is initialized.
    """
    global model, tokenizer, system_prompt, eos_token_ids

    system_prompt = (
        "You are a PII extraction assistant.\n"
        "Extract PII and return ONLY a JSON array (no prose) of objects with keys:\n"
        "- entity: one of [ADDRESS, NAME, EMAIL_ADDRESS, PHONE_NUMBER]\n"
        "- value: the extracted string\n"
        "If none, return [].\n"
        "Do not include any keys other than entity and value. Do not add commentary."
    )

    base_path = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(base_path, "model_output")

    print(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eos_token_ids = set()
    if model.config.eos_token_id is not None:
        if isinstance(model.config.eos_token_id, int):
            eos_token_ids.add(model.config.eos_token_id)
        else:
            eos_token_ids.update(model.config.eos_token_id)
    for tok in ["<|end|>", "<|eot_id|>"]:
        try:
            eid = tokenizer.convert_tokens_to_ids(tok)
            if eid is not None and eid != tokenizer.unk_token_id:
                eos_token_ids.add(eid)
        except Exception:
            pass

    print("Model and tokenizer loaded successfully.")

def extract_json_array(text: str):
    """
    Extract the first top-level JSON array from text.
    Returns '[]' if none found.
    """
    start = text.find("[")
    if start == -1:
        return "[]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break
    return "[]"

def normalize_schema(items):
    """
    Normalize output to the required schema.
    """
    mapped = []
    for it in items:
        if not isinstance(it, dict):
            continue
        entity = it.get("entity")
        value = it.get("value")
        if entity is None and value is None:
            if "email" in it: entity, value = "EMAIL_ADDRESS", it["email"]
            elif "address" in it: entity, value = "ADDRESS", it["address"]
            elif "name" in it: entity, value = "NAME", it["name"]
            elif "phone" in it or "phone_number" in it or "tel" in it:
                entity = "PHONE_NUMBER"
                value = it.get("phone") or it.get("phone_number") or it.get("tel")

        if not entity or value is None:
            continue

        entity = entity.upper()
        if entity not in {"ADDRESS", "NAME", "EMAIL_ADDRESS", "PHONE_NUMBER"}:
            continue
        
        mapped.append({"entity": entity, "value": str(value)})
    return mapped

def run(mini_batch):
    """
    Processes each file and writes the clean, raw model output to a JSONL file.
    """
    output_dir = os.environ.get("AZUREML_BI_OUTPUT_PATH")
    output_file_path = os.path.join(output_dir, "predictions.jsonl")

    processed_files = []

    with open(output_file_path, 'w', encoding="utf-8") as out_f:
        for file_path in mini_batch:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding="utf-8") as in_f:
                for line in in_f:
                    data = json.loads(line)
                    prompt_text = data["prompt"]

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]

                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt"
                    ).to(model.device)

                    input_token_length = inputs.shape[1]

                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            temperature=0.0,
                            eos_token_id=list(eos_token_ids) if eos_token_ids else tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    generated_tokens = outputs[0, input_token_length:]
                    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    json_text = extract_json_array(assistant_response)
                    try:
                        items = json.loads(json_text)
                        if not isinstance(items, list): items = []
                    except Exception:
                        items = []
                    items = normalize_schema(items)

                    result = {"prompt": prompt_text, "completion": json.dumps(items, ensure_ascii=False)}
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            processed_files.append({"file_processed": file_path, "status": "success"})

    return pd.DataFrame(processed_files)

init()