import torch
import numpy as np
from torch import nn

from typing import (
    Any,
    List,
    Dict,
    Optional,
    Mapping,
)

from neuralpredictors.layers.legacy import Gaussian2d
from nnfabrik.utility.nn_helpers import (
    set_random_seed, get_dims_for_loader_dict
)

from neuralpredictors.utils import get_module_output

from nnvision.models.readouts import MultipleFullGaussian2dJointReadout

from nnvision.models.utility import unpack_data_info
from nnvision.models.encoders import Encoder

from ptrnets.cores.cores import TaskDrivenCore2
from nnvision.models.cores import JointCore

from nnvision.utility.statedict_helpers import StateDictHandler
from nnvision.utility.readout_helpers import update_readout_bias_with_target_mean



def concat_task_cores_gauss_readout(
    dataloaders: Dict[str, Any],
    seed: int,
    input_channels: int = 1,
    model_name_1: str = "vgg19",  # begin of core_1 args
    layer_name_1: str = "features.10",
    pretrained_1: bool = True,
    bias_1: bool = False,
    final_batchnorm_1: bool = True,
    final_nonlinearity_1: bool = True,
    momentum_1: float = 0.1,
    fine_tune_1: bool = False,
    model_name_2: str = "vgg19",  # begin of core_2 args
    layer_name_2: str = "features.10",
    pretrained_2: bool = True,
    bias_2: bool = False,
    final_batchnorm_2: bool = True,
    final_nonlinearity_2: bool = True,
    momentum_2: float = 0.1,
    fine_tune_2: bool = False,
    pretrained_file: Optional[str] = None, # joint core params
    readout_split: str = "first",
    fields_to_initialize: List[str] = ["_features", "bias", "_mu", "sigma"],
    fine_tune_readout_1: bool = True,
    fine_tune_readout_2: bool = True,
    init_features_zeros: bool = False,
    init_mu_range: float = 0.4,    # readout args,
    init_sigma_range:float = 0.6,  
    readout_bias: bool = True,
    gamma_readout: float = 0.01,
    gauss_type: str = "isotropic",
    elu_offset: int = -1,
    data_info: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Concatenate the features of two task
    """
    
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )
    
    # Load pretrained readout weights (if any)
    readout_weights, core_weights = None, None
    if pretrained_file is not None:
        state_dict_handle = StateDictHandler(pretrained_file)
        readout_weights = state_dict_handle.readout_state_dict
        core_weights = state_dict_handle.core_state_dict

    # First core
    set_random_seed(seed)
    core_1 = TaskDrivenCore2(
        input_channels=core_input_channels,
        model_name=model_name_1,
        layer_name=layer_name_1,
        pretrained=pretrained_1,
        bias=bias_1,
        final_batchnorm=final_batchnorm_1,
        final_nonlinearity=final_nonlinearity_1,
        momentum=momentum_1,
        fine_tune=fine_tune_1,
    )
    set_random_seed(seed)
    core_1.initialize()
    
    # Load running statistics
    if readout_split == "first" and core_weights is not None:
        core_1.load_state_dict(core_weights, strict=False)        
    
    # Second core
    set_random_seed(seed)
    core_2 = TaskDrivenCore2(
        input_channels=core_input_channels,
        model_name=model_name_2,
        layer_name=layer_name_2,
        pretrained=pretrained_2,
        bias=bias_2,
        final_batchnorm=final_batchnorm_2,
        final_nonlinearity=final_nonlinearity_2,
        momentum=momentum_2,
        fine_tune=fine_tune_2,
    )
    set_random_seed(seed)
    core_2.initialize()
    
    # Load running statistics
    if readout_split == "second" and core_weights is not None:
        core_2.load_state_dict(core_weights, strict=False) 
    
    # Joint core
    core = JointCore(core_1, core_2)
    
    # Readout
    readout = MultipleFullGaussian2dJointReadout(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        state_dict = readout_weights,
        readout_split=readout_split,
        requires_grad_1=fine_tune_readout_1,
        requires_grad_2=fine_tune_readout_2,
        fields_to_initialize=fields_to_initialize,
        init_features_zeros=init_features_zeros,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=None,  # not relevant for monkey data
        grid_mean_predictor_type=None,
        source_grids=None,
        share_features=None,
        share_grid=None,
        shared_match_ids=None,
    )
    
    
    # Overwrite bias with target means
    if readout_bias and data_info is None:
        if readout_weights is None or "bias" not in fields_to_initialize:
            readout = update_readout_bias_with_target_mean(
                readout=readout,
                dataloaders=dataloaders,
            )
    
    # Static nonlinearity on readout outputs
    model = Encoder(core, readout, elu_offset=elu_offset)

    return model
