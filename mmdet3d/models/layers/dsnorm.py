# Ported from CMT: https://github.com/Jihan-Yang/CMT
# Original: Jihan Yang (2020), based on TransNorm (thuml).
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch.nn.parameter import Parameter


class DSNorm(nn.Module):
    """BatchNorm variant with separate running statistics per domain.

    Shared affine parameters (weight, bias) are learned jointly.
    Per-domain running stats prevent covariate shift when training alternates
    between source and target batches that have different feature distributions.

    Set the active domain via :func:`set_ds_source` / :func:`set_ds_target`
    applied with ``model.apply(...)`` before each forward pass.

    When ``use_dsnorm=True`` in ``mean_teacher_cfg``, the detector calls
    :meth:`convert_dsnorm` at construction time to replace every
    ``nn.BatchNorm{1,2}d`` in student and teacher with the appropriate subclass.
    Both source and target running buffers are seeded from the original BN stats,
    so pretrain-phase checkpoints (plain BN) warm-start correctly.

    Note:
        ``NaiveSyncBatchNorm{1,2}d`` subclasses ``BatchNorm{1,2}d`` and is
        therefore also converted. On single-GPU runs this is fine; the sync
        logic is simply replaced by the DSNorm path.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.domain_label = 0  # 0 = source, 1 = target

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))
            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_var_source.fill_(1)
            self.running_mean_target.zero_()
            self.running_var_target.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def set_domain_label(self, domain_label):
        self.domain_label = domain_label

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = self.momentum if self.momentum is not None else 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        running_mean = (self.running_mean_target
                        if self.domain_label else self.running_mean_source)
        running_var = (self.running_var_target
                       if self.domain_label else self.running_var_source)

        return F.batch_norm(
            input, running_mean, running_var,
            self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return ('{num_features}, eps={eps}, momentum={momentum}, '
                'affine={affine}, track_running_stats={track_running_stats}'
                ).format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            key = prefix + 'num_batches_tracked'
            if key not in state_dict:
                state_dict[key] = torch.tensor(0, dtype=torch.long)

        # Allow loading a plain-BN checkpoint: _source/_target suffix is
        # stripped to find the matching BN buffer name.
        local_name_params = itertools.chain(
            self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if ('source' in key or 'target' in key) and key not in state_dict:
                key = key[:-7]  # strip '_source' or '_target' (7 chars)
            if key in state_dict:
                input_param = state_dict[key]
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]
                if input_param.shape != param.shape:
                    error_msgs.append(
                        f'size mismatch for {key}: copying a param with shape '
                        f'{input_param.shape} from checkpoint, the shape in '
                        f'current model is {param.shape}.')
                    continue
                if isinstance(input_param, Parameter):
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append(
                        f'While copying "{key}": model shape {param.size()}, '
                        f'checkpoint shape {input_param.size()}.')
            elif strict:
                missing_keys.append(key)

    @classmethod
    def convert_dsnorm(cls, module):
        """Recursively replace all ``_BatchNorm`` layers with :class:`DSNorm`.

        Both ``running_mean_source`` and ``running_mean_target`` (and the var
        counterparts) are seeded from the original BN running stats.  This
        ensures that a pretrain-phase checkpoint (plain BN) warm-starts cleanly
        after conversion.

        Args:
            module (nn.Module): Root module to convert.

        Returns:
            nn.Module: Converted module tree.
        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            ds_cls = DSNorm1d if isinstance(module, nn.BatchNorm1d) else DSNorm2d
            module_output = ds_cls(
                module.num_features, module.eps, module.momentum,
                module.affine, module.track_running_stats)

            if module.affine:
                module_output.weight.data = module.weight.data.clone().detach()
                module_output.bias.data = module.bias.data.clone().detach()
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad

            if module.track_running_stats:
                module_output.running_mean_source.copy_(module.running_mean)
                module_output.running_mean_target.copy_(module.running_mean)
                module_output.running_var_source.copy_(module.running_var)
                module_output.running_var_target.copy_(module.running_var)
                module_output.num_batches_tracked.copy_(module.num_batches_tracked)

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_dsnorm(child))
        del module
        return module_output


@MODELS.register_module('DSNorm1d')
class DSNorm1d(DSNorm):
    """DSNorm for 2D or 3D inputs: (N, C) or (N, C, L)."""

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(
                f'expected 2D or 3D input (got {input.dim()}D input)')


@MODELS.register_module('DSNorm2d')
class DSNorm2d(DSNorm):
    """DSNorm for 4D inputs: (N, C, H, W)."""

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                f'expected 4D input (got {input.dim()}D input)')


def set_ds_source(m):
    """Set ``domain_label=0`` (source) on all DSNorm layers in a module."""
    if 'DSNorm' in m.__class__.__name__:
        m.set_domain_label(0)


def set_ds_target(m):
    """Set ``domain_label=1`` (target) on all DSNorm layers in a module."""
    if 'DSNorm' in m.__class__.__name__:
        m.set_domain_label(1)
