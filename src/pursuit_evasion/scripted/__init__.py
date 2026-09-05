from .attackers import MissileAttackers
from .base import (BaseController, Controller, NoisyController, RLController,
                   ZeroController)
from .defenders import GuardDefenders
from .evaders import FieldEvaders
from .pursuers import InterceptPursuers

__all__ = [
    "Controller", "BaseController", "RLController", "ZeroController", "NoisyController",
    "InterceptPursuers", "FieldEvaders", "MissileAttackers", "GuardDefenders",
]


def default_controllers(game_key: str = "tag"):
    """The scripted baseline pair (pursuer-slot, evader-slot) for a game."""
    if game_key == "tag":
        return InterceptPursuers(), FieldEvaders()
    return GuardDefenders(), MissileAttackers()
