# Main body borrowed from `https://github.com/AutoGPTQ/AutoGPTQ/blob/main/examples/benchmark/generation_speed.py`
# Adapted for our dynamic skipping and Wanda semi-structured model evaluation.

from model import DynamicSkippingMixtralSparseMoeBlockWrapper
import json
import time
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from typing import Dict, List, Optional, Union, Tuple

import torch
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, set_seed
from datasets import Dataset

import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralForCausalLM, MixtralDecoderLayer

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.sparse.semi_structured import _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG
from torch.utils.benchmark import Timer
import gc
import accelerate


logger = logging.getLogger(__name__)
SparseSemiStructuredTensor._FORCE_CUTLASS = True
random.seed(0)


def transform_dynamic_skip(model: MixtralForCausalLM):
    assert isinstance(model, MixtralForCausalLM)

    if not hasattr(model.config, 'betas'):
        logger.debug('Model does not perform dynamic skipping.')
        return
    else:
        logger.debug('Enabling dynamic skipping...')

    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
            layer.block_sparse_moe = DynamicSkippingMixtralSparseMoeBlockWrapper(
                layer.block_sparse_moe, float(model.config.betas[str(i)]))


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


class CustomizedMinNewTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        min_new_tokens: int = None,
        eos_token_id: int = None,
    ):
        self.eos_token_id = eos_token_id
        self.min_new_tokens = min_new_tokens or 0
        self.current_step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.current_step += 1

        if self._skip_process():
            return scores

        if any(each is not None for each in [self.eos_token_id]):
            banned_mask = torch.zeros_like(scores).to(scores.device)
            if self.eos_token_id and self.current_step <= self.min_new_tokens:
                banned_mask = self._fill_banned_mask(
                    input_ids, banned_mask, {1: [[self.eos_token_id]]})
            scores = scores.masked_fill(banned_mask.bool(), -float("inf"))

        return scores

    def _skip_process(self):
        if self.current_step > self.min_new_tokens:
            return True
        return False

    @staticmethod
    def _fill_banned_mask(
        input_ids: torch.LongTensor,
        banned_mask: torch.Tensor,
        len2words_ids: Dict[int, List[List[int]]]
    ):
        for token_len, token_ids in len2words_ids.items():
            if token_len == 1:
                banned_mask[..., list(chain(*token_ids))] = 1
            elif input_ids.shape[-1] < token_len - 1:
                continue
            else:
                token_ids = torch.LongTensor(token_ids).to(input_ids.device)
                hit_masks = torch.all(
                    token_ids[..., :-
                              1].unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
                    == input_ids[..., -(token_ids.shape[-1] - 1):].unsqueeze(1),
                    dim=-1
                )
                for idx in range(hit_masks.shape[0]):
                    selected_token_ids = torch.masked_select(
                        token_ids[..., -1], hit_masks[idx])
                    if len(selected_token_ids):
                        banned_mask[idx, selected_token_ids] = 1
        return banned_mask


def load_data(data_path, tokenizer, n_samples, max_new_tokens):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(
                tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(
                tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def transform_semi_structured_layers(module, name=''):
    if isinstance(module, nn.Linear):
        m, n = module.weight.shape
        min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
            module.weight.dtype
        ].min_rows
        min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
            module.weight.dtype
        ].min_cols
        if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
            logger.warn(f"Warning: {name}.shape {module.weight.shape} is not supported (thus will not be transformed)! "
                        f"Both dimensions must be larger or equal than and a multiple of ({min_rows}, {min_cols})")
        else:
            # import ipdb; ipdb.set_trace()
            module.to('cuda:0')
            with torch.no_grad():
                sparse_semi_structured_weight = torch.nn.Parameter(
                    to_sparse_semi_structured(module.weight))
                del module.weight
                module.weight = sparse_semi_structured_weight
            gc.collect()
            torch.cuda.empty_cache()
        return {name}
    res = set()
    for child_name, child in module.named_children():
        res.update(transform_semi_structured_layers(
            child, name=name + '.' + child_name if name != '' else child_name
        ))
    return res


def load_model_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    from_pretrained: bool = True,
    max_memory: Optional[dict] = None,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = False,
    parallelize: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path or model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if parallelize:
        model_kwargs = _get_accelerate_args(
            device_map_option="auto",
            max_memory_per_gpu=None,
            max_cpu_memory=None,
        )

    if from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            # quantize_config=BaseQuantizeConfig(),
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
    else:
        raise NotImplementedError

    # Evaluate wanda models
    if hasattr(model.config, 'wanda_sparsity_type') and ':' in model.config.wanda_sparsity_type:
        logger.info(
            f'Using wanda semi-strucutured model ({model.config.wanda_sparsity_type}). Now transforming to sparse semi-structured linear layers...')
        with torch.no_grad():
            transformed = transform_semi_structured_layers(model.model)
        model.to('cuda:0')
        for k in model.hf_device_map.keys():
            model.hf_device_map[k] = 0
        # import ipdb; ipdb.set_trace()
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
        logger.info(f'Transformed layers: {transformed}')

    return model, tokenizer


def benchmark_generation_speed(model, tokenizer, examples, generation_config):
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        torch.cuda.reset_accumulated_memory_stats(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.reset_max_memory_cached(device)
    generation_time_list = []
    num_generated_tokens_list = []
    progress_bar = tqdm(examples)
    for example in progress_bar:
        input_ids = example["input_ids"].to(model.device)

        start = time.time()
        # import ipdb; ipdb.set_trace()
        # model: MixtralForCausalLM
        outputs_ids = model.generate(
            input_ids=input_ids.unsqueeze(0),
            generation_config=generation_config,
            logits_processor=[
                CustomizedMinNewTokensLogitsProcessor(
                    generation_config.max_new_tokens, tokenizer.eos_token_id)
            ]
        )
        end = time.time()

        generation_time_list.append(end - start)
        num_generated_tokens = 0
        for output_ids in outputs_ids:
            num_generated_tokens += len(
                [
                    token_id for token_id in output_ids[len(input_ids):] if token_id != tokenizer.pad_token_id
                ]
            )
        num_generated_tokens_list.append(num_generated_tokens)

        progress_bar.set_postfix(
            num_tokens=num_generated_tokens_list[-1],
            time=generation_time_list[-1],
            speed=f"{num_generated_tokens_list[-1] / generation_time_list[-1]:.4f}tokens/s"
        )

    total_tokens = sum(num_generated_tokens_list)
    total_seconds = sum(generation_time_list)
    # import ipdb; ipdb.set_trace()
    logger.info(
        f"generated {total_tokens} tokens using {total_seconds} seconds, "
        f"generation speed: {total_tokens / total_seconds}tokens/s, "
        f"max gpu mem allocated: {[torch.cuda.max_memory_allocated(f'cuda:{i}') / 1024 / 1024 for i in range(torch.cuda.device_count())]}"
        f"max gpu mem reserved: {[torch.cuda.max_memory_reserved(f'cuda:{i}') / 1024 / 1024 for i in range(torch.cuda.device_count())]}"
        #  You can use memory_allocated() and max_memory_allocated() to monitor memory occupied by tensors
        #  and use memory_reserved() and max_memory_reserved() to monitor the total amount of memory managed by the caching allocator.
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="Mixtral-8x7B-v0.1")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--from_pretrained", type=bool, default=True)
    parser.add_argument("--quantize_config_save_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--per_gpu_max_memory", type=int, default=None)
    parser.add_argument("--cpu_max_memory", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(
                    torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    logger.info(f"max_memory: {max_memory}")

    logger.info("loading model and tokenizer")
    start = time.time()
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        from_pretrained=args.from_pretrained,
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )

    transform_dynamic_skip(model)

    end = time.time()
    logger.info(f"model and tokenizer loading time: {end - start:.4f}s")
    logger.info(f"model device map: {model.hf_device_map}")

    logger.info("loading data")
    examples = load_data(
        "data/alpaca_data_cleaned.json", tokenizer, args.num_samples, args.max_new_tokens
    )

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=args.do_sample,
        min_new_tokens=args.max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    logger.info(f"generation config: {generation_config.to_dict()}")

    logger.info(f"benchmark generation speed")
    benchmark_generation_speed(model, tokenizer, examples, generation_config)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
