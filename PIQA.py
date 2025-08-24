import torch
import numpy as np

def doc_to_choice(item):
    return [item['sol1'],item['sol2']]

def load_dataset():
    import json
    def load_dataset(input_filepath, label_filepath=None):
        examples = []
        with open(input_filepath, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        if label_filepath is not None:
            with open(label_filepath, "r", encoding="utf-8") as f:
                labels = [int(line.strip()) for line in f]

            for example, label in zip(examples, labels):
                example["label"] = label
        else:
            for example in examples:
                example["label"] = -1
        return examples
    data_path = "/cephfs/shared/wjj2/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl"
    dev_labels_file = "/cephfs/shared/wjj2/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst"
    return load_dataset(data_path, dev_labels_file)

def build_prompt(item):
    return f"Question: {item['goal']}\nAnswer:"

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
            if pred == item['label']:
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
    path = "/cephfs/shared/wjj2/physicaliqa-train-dev/physicaliqa-train-dev/train.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    sample_dataset = []

    for _ in range(args.sampled_size):
        try:
            sampled_rows = random.sample(data, 60)
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