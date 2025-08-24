from SVD import SVDModel

import importlib
import os
from typing import Any, Optional


def load_function(
        data_name: str,
        base_path: Optional[str] = None,
        function_name: Optional[str] = None,
        **kwargs: Any
) -> Any:
    if base_path is None:
        base_path = os.path.join(os.getcwd(), "DataLoad")

    if function_name is None:
        function_name = "load_dataset"

    spec = importlib.util.spec_from_file_location(data_name, base_path)

    try:
        module = importlib.util.module_from_spec(spec)
        if hasattr(module, function_name):
            dataset_func = getattr(module, function_name)
        else:
            print(f"Function {function_name} is not declared in {data_name}.py")
        return dataset_func(**kwargs)

    except Exception as e:
        raise RuntimeError(f"import {data_name} failed!")


def SVD4LLAMAMain(args, model, tokenizer):
    load_dataset = load_function(data_name=args.data, base_path="./DataLoad", function_name="load_dataset" )
    dataset = load_dataset(args.data)

    evalutate_total_dataset = load_function(data_name=args.data, base_path="./DataLoad", function_name="evalutate_total_dataset" )
    if args.not_eval_uncompressed_model:
        print("Evaluating uncompressed model")
        evalutate_total_dataset(dataset, model, tokenizer, args)
    sample_dataset_for_svd = load_function(data_name=args.data, base_path="./DataLoad", function_name="sample_dataset_for_svd" )
    sample_dataset = sample_dataset_for_svd(args)
    SVD_model = SVDModel(model, tokenizer, sample_dataset, args)

    if args.not_eval_compressed_model:
        print("Evaluating compressed model")
        evalutate_total_dataset(dataset, SVD_model, tokenizer, args)
