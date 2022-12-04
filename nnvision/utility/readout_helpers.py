from torch import nn
from typing import Any, Dict


def update_readout_bias_with_target_mean(
    readout: nn.ModuleDict,
    dataloaders: Dict[str, Any],
) -> nn.ModuleDict:
    
    for session, value in dataloaders.items():
        _, targets = next(iter(value))[:2]
        readout[session].bias.data = targets.mean(0)
    
    return readout
