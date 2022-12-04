from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Mapping

import os
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import nn

from nnfabrik.utility.dj_helpers import make_hash


DIR_PATH = "/tmp/readout_weights"


@dataclass
class StateDictHandler:
    file_name: str
    
    @property
    def path(self) -> str:
        return os.path.join(DIR_PATH, self.file_name)
    
    @property
    def path_exists(self) -> bool:
        return os.path.exists(self.path)
    
    @property
    def hash_name(self) -> str:
        return os.path.basename(self.path).split('.')[0]
    
    def load_state_dict(self) -> Mapping[str, Any]:
        if self.path_exists:
            return torch.load(self.path)
        
        from nnvision.tables.from_nnfabrik import TrainedModel
        trained_model_keys = TrainedModel().fetch("KEY")
        model_hash_names = list(map(make_hash, trained_model_keys))

        model_key = [
            key for key in trained_model_keys 
            if make_hash(key) == self.hash_name
        ]

        if len(model_key) == 0:
            raise ValueError(
                f"{self.hash_name} was not found in TrainedModel table"
            )
        
        # Fetch and download models
        model_state_dict_name = (
            TrainedModel.ModelStorage & model_key
        ).fetch1("model_state", download_path=DIR_PATH)

        assert self.path == model_state_dict_name
        return torch.load(self.path)
    
    @cached_property
    def readout_state_dict(self) -> Dict[str, Dict[str, Any]]:
        state_dict = self.load_state_dict()
        session_dict = defaultdict(dict)
        for k, v in state_dict.items():
            if k.startswith('readout'):
                k = k.replace("readout.", "")
                session, attribute = k.split('.')
                session_dict[session][attribute] = v
        return dict(session_dict)
    
    @cached_property
    def core_state_dict(self) -> Dict[str, Dict[str, Any]]:
        state_dict = self.load_state_dict()
        session_dict = defaultdict(dict)
        for k, v in state_dict.items():
            if k.startswith('core.'):
                k = k.replace("core.", "")
                session_dict[k] = v
        return dict(session_dict)
