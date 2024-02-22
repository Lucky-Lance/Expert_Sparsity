from . import expert_pruning
from . import dynamic_skipping


METHODS = {
    'layerwise_pruning': expert_pruning.layerwise_pruning,
    'progressive_pruning': expert_pruning.progressive_pruning,
    'dynamic_skipping': dynamic_skipping.dynamic_skipping,
}
