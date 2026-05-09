from typing import Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class MeanTeacherHook(Hook):
    """Drives the EMA update for :class:`MeanTeacher3DDetector`.

    After every ``interval`` training iterations, calls ``model.ema_update()``
    to update the teacher's parameters as an exponential moving average of the
    student.  The momentum itself lives in ``model.mean_teacher_cfg`` so that
    it can be set once in the detector config rather than split across two
    config files.

    Args:
        interval (int): EMA update frequency (in iterations). Default: 1.
    """

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval

    def before_train(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher'), \
            'MeanTeacherHook requires model.teacher'
        assert hasattr(model, 'student'), \
            'MeanTeacherHook requires model.student'
        assert hasattr(model, 'ema_update'), \
            'MeanTeacherHook requires model.ema_update()'

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int = None,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.ema_update()
