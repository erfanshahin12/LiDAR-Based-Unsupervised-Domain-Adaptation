# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze every parameter whose name does NOT contain any of
    ``name_substrings``.

    Submodules whose direct parameters are all frozen are also put into
    ``.eval()`` mode at the start of training and at every epoch start,
    so BatchNorm running stats and Dropout become deterministic.

    Args:
        name_substrings: substrings that, when present anywhere in a
            parameter's qualified name, mark that parameter as
            trainable. All other parameters are frozen.
    """

    priority = 'HIGHEST'

    def __init__(self, name_substrings: Sequence[str]) -> None:
        assert len(name_substrings) > 0, \
            'name_substrings must contain at least one substring'
        self.name_substrings = list(name_substrings)

    def _is_trainable(self, name: str) -> bool:
        return any(sub in name for sub in self.name_substrings)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    def before_train(self, runner) -> None:
        model = self._unwrap(runner.model)
        trainable, frozen = [], []
        for n, p in model.named_parameters():
            if self._is_trainable(n):
                p.requires_grad_(True)
                trainable.append(n)
            else:
                p.requires_grad_(False)
                frozen.append(n)

        logger = MMLogger.get_current_instance()
        logger.info(
            f'[FreezeExceptHook] trainable params ({len(trainable)}): '
            + ', '.join(trainable))
        logger.info(
            f'[FreezeExceptHook] frozen params: {len(frozen)} '
            f'(first 5) -> {frozen[:5]}')

        self._apply_eval(model)

    def before_train_epoch(self, runner) -> None:
        # runner.model.train() is called every epoch; re-apply eval()
        # to fully-frozen submodules so BN stats stay fixed.
        self._apply_eval(self._unwrap(runner.model))

    def _apply_eval(self, model) -> None:
        for _, module in model.named_modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            if all(not p.requires_grad for p in params):
                module.eval()
