from memory import memset_zero, memset
from sys.info import simd_width_of
from algorithm.functional import vectorize

struct Series[
        _dtype: DType, _size: Int, _name: Optional[String] = None
](
    AnyType & Writable & Stringable
):
    """
    A 1D labeled array holding data of a single type. The core data column.
    """
    alias simd_width: Int = simd_width_of[_dtype]()
    alias IDX_WIDTH: Int = Self._digits(_size)
    var _data: UnsafePointer[Scalar[_dtype]]
    var _own_data: Bool

    @always_inline("nodebug")
    fn __init__(out self):
        self._data = UnsafePointer[Scalar[_dtype]].alloc(_size)
        # memset_zero(self._data, _size)
        memset(self._data, 1, _size)
        self._own_data = True

    @always_inline("nodebug")
    fn __init__(out self, data: UnsafePointer[Scalar[_dtype]]):
        self._data = data
        self._own_data = False

    fn __del__(deinit self):
        if self._own_data:
            self._data.free()

    fn __str__(self) -> String:
        return(self._construct_output_string())

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self._construct_output_string())

    fn head(self, n: Int = 5) -> String:
        return self._construct_output_string(n)

    fn tail(self, n: Int = 5) -> String:
        return self._construct_output_string(n, True)

    # ------------------------------------------------------- #
    # ----------------------Operations ---------------------- #
    # ------------------------------------------------------- #

    fn __add__[count: Int, //](self, *rhs: Series[_dtype, _size]) -> Series[_dtype, _size]:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            @parameter
            for rh in range(count):
                b = rhs[i]._data.load[width=simd_width](i)
                p.store(i, a + b)
        vectorize[closure, Self.simd_width](_size)
        return Series[_dtype, _size](p)

    # ------------------------------------------------------- #
    # ----------------------- Helpers ----------------------- #
    # ------------------------------------------------------- #

    @parameter
    @staticmethod
    fn _digits(n: Int) -> Int:
        """
        Helper function to calculate the spaces between index, | and value
        """
        var d = 1
        var x = n - 1
        while x >= 10:
            x //= 10
            d += 1
        return d

    @always_inline("nodebug")
    fn _construct_output_string(self, n: Optional[Int] = None, tail: Bool = False) -> String:
        var str_data: String
        if self._name:
            str_data = "name='" + self._name.value() + "'" +
            " shape=(" + String(_size) + ",) " + String(_dtype) + "\n"
        else:
            str_data = "name='' shape=(" + String(_size) + ",) " + String(_dtype) + "\n"
        if n and tail:
            for i in range(_size-n.value(),_size):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str_data += " " * pad
                str_data += (idx_s + " | " + String(self._data[i]) + "\n")
        elif n :
            for i in range(n.value()):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str_data += " " * pad
                str_data += (idx_s + " | " + String(self._data[i]) + "\n")
        else:
            for i in range(_size):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str_data += " " * pad
                str_data += (idx_s + " | " + String(self._data[i]) + "\n")
        return str_data

fn main():
    var s = Series[DType.float32, 60]()
