import argparse
import torch
import numpy as np
import os
import re
import torch.nn.functional as F


def doc_to_choice(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices

def load_dataset():
    import json
    data_path = "/cephfs/shared/wjj2/MathQA/test.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_output(model, tokenizer, prompt, choices, args):
    probs=[]
    for choice in choices:
        probs.append(loglikelihood(model, tokenizer, prompt, choice, args))
    pred = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}[np.argmax(probs)]
    print(pred, probs)
    return pred


def loglikelihood(model, tokenizer, prompt, choice, args):
    input_ids = tokenizer.encode(prompt + " " + choice, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    continuation_start = len(tokenizer.encode(prompt))
    continuation_ids = input_ids[:,continuation_start:]
    selected_log_probs = log_probs[:, continuation_start - 1:-1, :].gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = selected_log_probs.sum().item()
    return total_log_prob

def evalutate_tatal_dataset(dataset, model, tokenizer, args):
    corr = 0
    total = len(dataset)

    for item in dataset:
        try:
            prompt = build_prompt(item)
            choices = doc_to_choice(item)
            if args.use_logits_to_generete_output:
                pred = generate_output(model, tokenizer, prompt, choices, args)
            else:
                pass
            if pred == item["correct"]:
                corr += 1
        except Exception as e:
            print(f"\nError processing sample: {str(e)}")
            import traceback
            traceback.print_exc()
    accuracy = corr / total if total > 0 else 0.0
    print(f'Correct-{corr}, Total-{total}, Accuracy-{100 * accuracy:.4f}%')

def build_prompt(item):
    prompt = f"Question: {item['Problem']}\nAnswer:"
    return prompt

def sample_dataset_for_svd(args):
    import json
    import random
    path = "/cephfs/shared/wjj2/MathQA/train.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sample_dataset = []

    for _ in range(args.sampled_size):
        try:
            sampled_rows = random.sample(data, 30)
            prompts = []
            for item in sampled_rows:
                prompt = build_prompt(item)
                prompts.append(prompt)
            combined_prompt = "\n".join(prompts)
            sample_dataset.append(combined_prompt)

        except Exception as e:
            print(f"\nError processing sample: {str(e)}")
            import traceback
            traceback.print_exc()

    return sample_dataset