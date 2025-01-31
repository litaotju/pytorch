# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from torch.distributed.optim import _apply_optimizer_in_backward

# TODO (rohan-varma): Add FSDP & DDP tests once supported

def _validate_params(params_list, fn):
    ref_params = params_list[0]
    for param_list in params_list[1:]:
        for p1, p2 in zip(ref_params, param_list):
            fn(p1, p2)


class ApplyOverlappedOptimizerTest(unittest.TestCase):

    def _run_training_loop_and_validate(self, inp, models, optimizers):
        for i in range(6):
            for model in models:
                model(inp).sum().backward()
            for opt in optimizers:
                opt.step()

            with self.subTest(i):
                _validate_params(
                    [model.parameters() for model in models],
                    torch.testing.assert_allclose,
                )

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

    def _test_apply_optimizer_in_backward(self, share_params) -> None:
        weight_optimizer_kwargs = {"lr": 1.0}
        bias_optimizer_kwargs = {"lr": 0.5}
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        if share_params:
            model[0].weight = model[1].weight

        # Use different optimizers for weights & biases.
        weights = [m.weight for m in model]
        biases = [m.bias for m in model]
        optim_weight = torch.optim.SGD(weights, **weight_optimizer_kwargs)
        optim_bias = torch.optim.SGD(biases, **bias_optimizer_kwargs)
        model_with_opt_in_bwd = deepcopy(model)

        # Apply different optimizer in backwards for weights and biases.
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.weight for m in model_with_opt_in_bwd],
            optimizer_kwargs=weight_optimizer_kwargs
        )

        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.bias for m in model_with_opt_in_bwd],
            optimizer_kwargs=bias_optimizer_kwargs
        )

        _validate_params(
            [
                model.parameters(),
                model_with_opt_in_bwd.parameters(),
            ],
            torch.testing.assert_allclose,
        )

        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [optim_weight, optim_bias],
        )

    def test_apply_optimizer_in_backward(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=False)

    def test_apply_optimizer_in_backward_shared_params(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=True)

    def test_multiple_optim_for_params(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        opt_0_kwargs = {"lr": 0.03}
        opt_1_kwargs = {"lr": 0.01}
        opt_0 = torch.optim.SGD(model.parameters(), **opt_0_kwargs)
        opt_1 = torch.optim.SGD(model.parameters(), **opt_1_kwargs)
        model_with_opt_in_bwd = deepcopy(model)
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_0_kwargs,
        )
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_1_kwargs,
        )
        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [opt_0, opt_1],
        )
