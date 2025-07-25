from .hook import Hook
from ...nets.vit.dualpt import DualPrompt
from ...nets.vit.vpt import VPT


class FreezeHook(Hook):
    def before_run(self, algorithm):
        if isinstance(algorithm.model, VPT) or isinstance(algorithm.model, DualPrompt):
            algorithm.model.freeze()
