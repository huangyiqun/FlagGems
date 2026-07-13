import pytest
import torch

from . import base, consts, utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


def cumsum_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(inp)
    yield inp, -1, {"out": out}


@pytest.mark.cumsum
def test_cumsum():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumsum",
        torch_op=torch.cumsum,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()


@pytest.mark.cumsum_out
def test_cumsum_out():
    bench = base.GenericBenchmark2DOnly(
        input_fn=cumsum_out_input_fn,
        op_name="cumsum_out",
        torch_op=torch.cumsum,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
