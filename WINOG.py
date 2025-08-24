import torch
import numpy as np

def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_")
    return doc["sentence"][:idx].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [opt + doc["sentence"][idx+1:] for opt in options  ]


def load_dataset():
    import json
    data_path = "/cephfs/shared/wjj2/winogrande_1.1/winogrande_1.1/dev.jsonl"
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_prompt(item):
    return doc_to_target(item)

def generate_output(model, tokenizer, prompt, choices, args):
    probs = []
    for choice in choices:
        probs.append(loglikelihood(model, tokenizer, prompt, choice, args))
    pred = np.argmax(probs)
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
            if pred == doc_to_text(item):
                corr += 1
        except Exception as e:
            print(f"\nError processing sample: {str(e)}")
            import traceback
            traceback.print_exc()
    accuracy = corr / total if total > 0 else 0.0
    print(f'Correct-{corr}, Total-{total}, Accuracy-{100 * accuracy:.4f}%')


def sample_dataset_for_svd(args):
    import json
    import random
    path = "/cephfs/shared/wjj2/winogrande_1.1/winogrande_1.1/train_l.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
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