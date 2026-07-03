from typing import Generator

import pytest
import torch

from .attri_util import BOOL_DTYPES, DEFAULT_METRICS, FLOAT_DTYPES, INT_DTYPES
from .performance_utils import Benchmark, generate_tensor_input


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        shape2 = list(args[0].shape)
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        # Arithmetic operations
        pytest.param("add", torch.add, FLOAT_DTYPES, marks=pytest.mark.add),
        pytest.param("div", torch.div, FLOAT_DTYPES, marks=[pytest.mark.div, pytest.mark.true_divide]),
        pytest.param("mul", torch.mul, FLOAT_DTYPES, marks=pytest.mark.mul),
        pytest.param("pow", torch.pow, FLOAT_DTYPES, marks=[pytest.mark.pow, pytest.mark.pow_tensor_tensor]),
        pytest.param("sub", torch.sub, FLOAT_DTYPES, marks=pytest.mark.sub),
        pytest.param("floor_divide", torch.floor_divide, INT_DTYPES, marks=pytest.mark.floor_divide),
        pytest.param("remainder", torch.remainder, INT_DTYPES, marks=pytest.mark.remainder),
        pytest.param("rsub", torch.rsub, FLOAT_DTYPES, marks=pytest.mark.rsub),
        pytest.param("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES, marks=pytest.mark.logical_or),
        pytest.param("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES, marks=pytest.mark.logical_and),
        pytest.param("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES, marks=pytest.mark.logical_xor),
        # Comparison operations
        pytest.param("eq", torch.eq, FLOAT_DTYPES, marks=pytest.mark.eq),
        pytest.param("ge", torch.ge, FLOAT_DTYPES, marks=pytest.mark.ge),
        pytest.param("gt", torch.gt, FLOAT_DTYPES, marks=pytest.mark.gt),
        pytest.param("le", torch.le, FLOAT_DTYPES, marks=pytest.mark.le),
        pytest.param("lt", torch.lt, FLOAT_DTYPES, marks=pytest.mark.lt),
        pytest.param("ne", torch.ne, FLOAT_DTYPES, marks=pytest.mark.ne),
        # Minimum and maximum operations
        pytest.param("maximum", torch.maximum, FLOAT_DTYPES, marks=pytest.mark.maximum),
        pytest.param("minimum", torch.minimum, FLOAT_DTYPES, marks=pytest.mark.minimum),
        # Bitwise operations
        pytest.param("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES, marks=[pytest.mark.bitwise_and, pytest.mark.bitwise_and_tensor]),
        pytest.param("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES, marks=[pytest.mark.bitwise_or, pytest.mark.bitwise_or_tensor]),
        pytest.param("or_", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES, marks=pytest.mark.or_),
        # Numerical Checks
        pytest.param("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES, marks=pytest.mark.isclose),
        pytest.param("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES, marks=pytest.mark.allclose),
    ],
)
def test_general_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()
