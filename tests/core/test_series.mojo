from grizzlies.core.series import Series
from testing import *

def test_constructors():
    s = Series[DType.float32, 3]()
    assert_equal(s._data[0].dtype, DType.float32)

def test_addition():
    s = Series[DType.float32, 3]()
    s += 3
    assert_equal(s[0], 3)
