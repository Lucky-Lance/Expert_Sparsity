# Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

Official Pytorch implementation of the expert pruning and dynamic skipping methods as presented in:

**Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models (ACL 2024, main)**</br>
*Xudong Lu\*,Qi Liu\*, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li* (\* indicates equal contribution)</br>
CUHK MMlab, Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory</br>
[Paper](https://arxiv.org/abs/2402.14800)

## Installation
Step 1: Create a new conda environment:
```bash
conda create -n env_name python=3.10
conda activate env_name
```
Step 2: Install relevant packages

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.36.2 accelerate datasets fire tqdm
```

## Dataset Preparation
1. C4: Please download first part of the C4 training data `c4-train.00000-of-01024.json` from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz).
2. MATH: You can use our pre-built calibration set in `./data/math_pretrain_style.json`. To reproduce our construction, please download the training set of [MATH](https://github.com/hendrycks/math) and use our [script](data/math_calib_construction.py).
3. Alpaca: In our project, generation speed is benchmarked using Alpaca dataset. Please download `alpaca_data_cleaned.json` from [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).
4. Finally, please organize the calibration datasets as follows.
```
./data
|-- __init__.py
|-- alpaca_data_cleaned.json
|-- build.py
|-- c4-train.00000-of-01024.json
|-- dataset.py
|-- math_calib_construction.py
`-- math_pretrain_style.json
```

## Pruning
Usage:
```
python main.py [-h] --method {layerwise_pruning,progressive_pruning,dynamic_skipping} [--r R] --calib_set {c4,math} --model_path MODEL_PATH [--output_path OUTPUT_PATH] [--max_block_size MAX_BLOCK_SIZE] [--n_blocks_for_stat N_BLOCKS_FOR_STAT] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--seed SEED] [--use_flash_attention_2]
```
Options:
- `-h, --help`: show this help message and exit
- `--method {layerwise_pruning,progressive_pruning,dynamic_skipping}`: Supported pruning methods: Layerwise Pruning, Progressive Pruning, Dynamic Skipping
- `--r R`: Number of experts to preserve
- `--calib_set {c4,math}`: Supported calibration datasets: C4, MATH
- `--model_path MODEL_PATH`: Path to model to prune
- `--output_path OUTPUT_PATH`: Output path (pruned model, pruning results, etc.)
- `--max_block_size MAX_BLOCK_SIZE`: Maximal sequence length of each sample in calibration set
- `--n_blocks_for_stat N_BLOCKS_FOR_STAT`: Number of sequences in calibration set. If set to 0 or negative, the whole dataset will be used
- `--batch_size BATCH_SIZE`: Batch size for model inference
- `--num_workers NUM_WORKERS`: Number of workers in dataloader
- `--seed SEED`: Random seed for reproduction
- `--use_flash_attention_2`: If set, Flash Attention 2 will be used

### One Example for Expert Pruning
You can can perform expert pruning by running:
```
python main.py --method layerwise_pruning --r 6 --calib_set c4 --model_path Mixtral-8x7B-v0.1 --output_path ./output/
```

**Note**: To integrate dynamic skipping with other methods, please modify your installed `transformers/models/mixtral/modeling_mixtral.py` as [`model/modeling_mixtral.py`](model/modeling_mixtral.py)

## Evaluation

### LM Harness Evaluation
We use the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/2a47159caff00135b026f724ace2a2011f3c7621) (commit 2a47159) framework to evaluate the performance of pruned LLMs. 
The command we use for LM Harness evaluation is as follows:
```bash
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    --ipex \
    -m lm_eval --model hf \
        --model_args pretrained=model_path,dtype=bfloat16,parallelize=True \
        --tasks boolq,rte,arc_challenge,arc_easy,hellaswag,winogrande,openbookqa,gsm8k,mmlu\
        --batch_size 16
```

### Speedup Evaluation
We use [eval/benchmark_speed.py](eval/benchmark_speed.py) to evaluate the speedup of pruned LLMs.
The command we use for speedup evaluation is as follows:

```bash
python ./benchmark_speed.py --num_samples 50 --model_name_or_path Mixtral-8x7B-v0.1
```

For finer-grained options, please refer to the script.

To benchmark generation speed for semi-structured pruned models, please add a new item `wanda_sparsity_type` into `model.config` (e.g., 2:4).

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions

Feel free to discuss papers/code with us through issues/emails!

- Xudong Lu: <a href="luxudong@link.cuhk.edu.hk">luxudong@link.cuhk.edu.hk</a> 
- Qi Liu: <a href="purewhite@sjtu.edu.cn">purewhite@sjtu.edu.cn</a> 

## Citation

If you find our paper and code useful in your research, please cite

```
@misc{lu2024experts,
      title={Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models}, 
      author={Xudong Lu and Qi Liu and Yuhui Xu and Aojun Zhou and Siyuan Huang and Bo Zhang and Junchi Yan and Hongsheng Li},
      year={2024},
      eprint={2402.14800},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
