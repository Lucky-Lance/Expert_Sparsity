from tqdm import tqdm
from argparse import Namespace
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

from model import PrunableMixtralSparseMoeBlockWrapper


logger = logging.getLogger(__name__)


def dynamic_skipping(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe)
        layer.block_sparse_moe.cache_logits = True
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    res_median = {}
    res_mean = {}

    for layer_idx in range(len(model.model.layers)):
        b = model.model.layers[layer_idx].block_sparse_moe
        b.cache_space.prepare_for_loader()
        dataloader = torch.utils.data.DataLoader(
            b.cache_space,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info(len(dataloader))

        ana_list = []
        for i, (router_logits, X, Z) in enumerate(dataloader):
            routing_weights = F.softmax(
                router_logits, dim=-1, dtype=torch.float).view(-1, b.model.num_experts)
            for j in range(len(routing_weights)):
                sorted_weights, sort_indices = torch.sort(
                    routing_weights[j], descending=True)
                ana_list.append(float(sorted_weights[1] / sorted_weights[0]))

        median = np.median(ana_list)
        mean = np.mean(ana_list)
        logger.info(f'layer {layer_idx} | mean: {mean}, median: {median}')
        res_median[str(layer_idx)] = median
        res_mean[str(layer_idx)] = mean

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.config.betas = res_median
    return model, (res_median, res_mean)
