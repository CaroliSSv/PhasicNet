##Collection file for the meta-system dataclasses
from __future__ import annotations
from dataclasses import dataclass, asdict, replace, field
from typing import Any, Dict, Optional, Literal, Tuple
import logging
from utils import _capitalize_strip_name
logger = logging.getLogger(__name__)



##############################################################################
 ############################## NetMeta #####################################

@dataclass(slots=True, frozen=True)
class NetMeta:
    spec_id: str | None = None
    spec_name: str | None=None
    network_id: str | None = None
    network_name: str|None = None
    post_offset: bool|None = None
    globalseed: int | None = None
    rng_spec: Dict | None = None


##############################################################################
 ############################## RunMeta #####################################


@dataclass(slots=True, frozen=True)
class RunMeta:
    run_name:str= None ## 1 run = 1 history same name used (no seperate history name)
    run_id:str = None          
    cfg_name:str = None
    cfg_id:str = None            
    derived_cfg_id:str = None    ##hash of the cfg ACTUALLY used (derived from modules given to Run, NOT from cfgs)
    network_name: str = None
    network_id: str = None
    dataset_name: str = None     
    dataset_id: str = None
    dataset_id_n_samples: int = None
    globalseed:int = None
    rng_spec:Dict = None

