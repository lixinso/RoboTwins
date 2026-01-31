from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

Observation = Dict[str, Any]


class SimEnv(ABC):
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = "human") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        return None
