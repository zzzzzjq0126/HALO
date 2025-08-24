# README

### Example

If you want to compress on the PIQA dataset or evaluate the compression results while maintaining a compression ratio of 0.2, and save the difference matrix with a 5% proportion. You can achieve this through the following methods:

```
python Main.py
--data PIQA  
--compress_ratio 0.20
```

If you don't want to use the difference matrix, you can run the code as follows:

```
python Main.py
--not_delta
```

If you want to change the storage location of the compressed model, you can modify the parameter `data_save_path` as shown below:

```
python Main.py
--data_save_path ./models/save
--data PIQA  
--compress_ratio 0.20
--sparsity_ratio 0.05
```

The compressed model will be stored at the `PIQA_20_5` under this path

The code will automatically evaluate the target data before and after compression. If you want to skip the evaluation step, you can use the following code:

```
python Main.py
--not_eval_uncompressed_model
--not_eval_compressed_model
```

More information can be viewed through the comments in the `Main.py` file.

