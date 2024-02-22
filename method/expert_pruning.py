from tqdm import tqdm
from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

from model import PrunableMixtralSparseMoeBlockWrapper


logger = logging.getLogger(__name__)


def layerwise_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    logger.info('Moving whole model to cpu...')
    model.to('cpu')
    torch.cuda.empty_cache()

    global_loss_history = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        b = layer.block_sparse_moe
        if not hasattr(b, 'cache_space'):
            continue
        if l < 16:
            b.to('cuda:0')
        else:
            b.to('cuda:1')
        loss_history = b.enumerate()
        global_loss_history[l] = loss_history
        b.prune()
        b.to('cpu')

    logger.info('Merging & saving...')
    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )


def progressive_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate()
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )
