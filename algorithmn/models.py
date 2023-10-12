from dataclasses import dataclass, field
from typing import Any, Dict
from torch import FloatTensor


@dataclass
class LocalTrainResult:
    weights: Dict[str, Any] = field(default_factory=dict)
    loss_map: Dict[str, float | FloatTensor] = field(default_factory=dict)
    acc_map: Dict[str, float | FloatTensor] = field(default_factory=dict)

@dataclass
class GlobalTrainResult:
    loss_map: Dict[str, float | FloatTensor] = field(default_factory=dict)
    acc_map: Dict[str, float | FloatTensor] = field(default_factory=dict)