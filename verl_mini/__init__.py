# Copyright 2024 - verl_mini reproduction
# A simplified reproduction of ByteDance's verl framework
# for understanding RLHF training concepts

from .protocol import DataProto, DataProtoItem
from .base_config import BaseConfig

__version__ = "0.1.0"
__all__ = ["DataProto", "DataProtoItem", "BaseConfig", "__version__"]
