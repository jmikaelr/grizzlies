from memory import memset_zero
from sys.info import simd_width_of
from algorithm.functional import vectorize

struct Series[
        _dtype: DType, _size: Int, _name: Optional[String]=None
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
        memset_zero(self._data, _size) # Have to memset otherwise it's not initialised?
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

    # ------------------- Series & Scalars ------------------ #

    fn __add__(self, rhs: Scalar[_dtype]) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            p.store[width=simd_width](i, a + rhs)
        vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __iadd__(self, rhs: Scalar[_dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            self._data.store(i, a + rhs)
        vectorize[closure, Self.simd_width](_size)

    fn __sub__(self, rhs: Scalar[_dtype]) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            p.store(i, a - rhs)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __isub__(self, rhs: Scalar[_dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            self._data.store(i, a - rhs)
        vectorize[closure, Self.simd_width](_size)

    fn __mul__(self, rhs: Scalar[_dtype]) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            p.store(i, a * rhs)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __imul__(self, rhs: Scalar[_dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            self._data.store(i, a * rhs)
        vectorize[closure, Self.simd_width](_size)

    fn __div__(self, rhs: Scalar[_dtype]) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            p.store(i, a / rhs)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __idiv__(self, rhs: Scalar[_dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            self._data.store(i, a / rhs)
        vectorize[closure, Self.simd_width](_size)

    # ------------------- Series & Series ------------------- #

    fn __add__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            p.store(i, a + b)
        vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __iadd__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            self._data.store(i, a + b)
        vectorize[closure, Self.simd_width](_size)

    fn __sub__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            p.store(i, a - b)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __isub__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            self._data.store(i, a - b)
        vectorize[closure, Self.simd_width](_size)

    fn __mul__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            p.store(i, a * b)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __imul__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            self._data.store(i, a * b)
        vectorize[closure, Self.simd_width](_size)

    fn __div__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[_dtype]].alloc(_size)
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            p.store(i, a / b)
            vectorize[closure, Self.simd_width](_size)
        return Self(p)

    fn __idiv__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._data.load[width=simd_width](i)
            b = rhs._data.load[width=simd_width](i)
            self._data.store(i, a / b)
        vectorize[closure, Self.simd_width](_size)

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
    var s = Series[DType.float32, 40320942]()
    t = s + 3
    print(t.tail())
