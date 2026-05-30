"""
Simplified reproduction of verl's BaseConfig.
Provides a dict-like interface for dataclass configurations with frozen fields.
"""

import collections
from dataclasses import dataclass, fields, FrozenInstanceError
from typing import Any


@dataclass
class BaseConfig(collections.abc.Mapping):
    """
    BaseConfig provides dict-like interface for a dataclass config.
    
    By default, all fields are immutable unless specified in _mutable_fields.
    This class implements the Mapping ABC, allowing instances to be used like dicts.
    """
    
    _mutable_fields = set()
    _target_: str = ""
    
    def __setattr__(self, name: str, value: Any):
        """Set attribute value, checking if field is mutable."""
        if name in self.__dict__ and name not in getattr(self, "_mutable_fields", set()):
            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")
        super().__setattr__(name, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if not found."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access with []."""
        return getattr(self, key)
    
    def __iter__(self):
        """Iterate over field names."""
        for f in fields(self):
            yield f.name
    
    def __len__(self) -> int:
        """Return number of fields."""
        return len(fields(self))
    
    def to_dict(self) -> dict:
        """Convert to regular dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
