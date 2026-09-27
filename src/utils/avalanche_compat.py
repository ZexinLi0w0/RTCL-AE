"""
Compatibility shims for the pinned Avalanche build (0.6.0a, SHA eb075be).

Avalanche's own `ReplayPlugin.after_training_exp` still calls the deprecated
`ExemplarsBuffer.update()`, which emits a DeprecationWarning on every experience
and only forwards to `post_adapt(strategy, strategy.experience)`. We call
`post_adapt` directly, so buffer updates are unchanged.

Importing this module applies the shim; it is idempotent.
"""

from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ExemplarsBuffer

def _replay_after_training_exp(self, strategy, **kwargs):
    policy = self.storage_policy
    if type(policy).update is not ExemplarsBuffer.update:
        # Policies that still override update() (e.g. _ParametricSingleBuffer) keep their own logic.
        policy.update(strategy, **kwargs)
    else:
        policy.post_adapt(strategy, strategy.experience)

def apply():
    ReplayPlugin.after_training_exp = _replay_after_training_exp

apply()
