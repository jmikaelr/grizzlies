from memory import memset_zero, memset
from sys.info import simd_width_of
from algorithm.functional import vectorize
from random import randn, rand


fn _series_construction_checks[size: Int]():
    constrained[size >= 0, "number of elements in `Series` must be >= 0"]()


struct PrintOptions:
    ...


struct SeriesView[dtype: DType](Movable & Stringable & Writable):
    var base: UnsafePointer[Scalar[dtype]]
    var len: Int
    var offset: Int
    var stride: Int

    fn __init__(
        out self,
        base: UnsafePointer[Scalar[dtype]],
        offset: Int = 0,
        len: Int = 0,
        stride: Int = 1,
    ):
        self.base = base
        self.len = len
        self.offset = offset
        self.stride = stride

    fn debug_print(self):
        print(
            "base",
            self.base,
            "\nlen",
            self.len,
            "\noffset",
            self.offset,
            "\nstride",
            self.stride,
        )

    fn _construct_view(self) -> String:
        var str: String = "SeriesView("
        for i in range(self.len + 1):
            addr = self.base + i * self.stride
            print(addr)
            str += String(self.base[self.offset + i * self.stride])
            if i != self.len:
                str += ", "
        str += ")"
        return str

    fn __str__(self) -> String:
        return self._construct_view()

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self._construct_view())

    fn values(self) -> List[Scalar[dtype]]:
        """
        Creates a copy of the data and return it.
        """
        n = self.len
        var out = List[Scalar[dtype]](capacity=n)
        for i in range(n):
            pos = self.offset + i * self.stride
            out.insert(i, self.base[pos])

        return out^


struct Series[
    dtype: DType,
    size: Int,
    name: Optional[String] = None,
    address_space: AddressSpace = AddressSpace.GENERIC,  # AddressSpace(0)
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin],
](Writable & Stringable & Sized):

    """
    A 1D labeled array holding data of a single type. The core data column.
    """

    alias SIMD_WIDTH: Int = simd_width_of[dtype]()
    alias IDX_WIDTH: Int = Self._digits(size)
    var _ptr: UnsafePointer[Scalar[dtype]]
    var _own_ptr: Bool

    @always_inline
    fn __init__(out self):
        _series_construction_checks[size]()

        constrained[
            False,
            (
                "Initialize with the constructor methods "
                "or pass the keyword argument 'uninitialized=True'."
            ),
        ]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__(out self, *, uninitialized: Bool):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    fn __init__(out self, data: UnsafePointer[Scalar[dtype]]):
        _series_construction_checks[size]()
        self._ptr = data
        self._own_ptr = True

    fn __del__(deinit self):
        @parameter
        if not Bool(dtype.__del__is_trivial):

            @parameter
            for idx in range(size):
                var ptr = self._ptr + idx
                ptr.destroy_pointee()

    fn __len__(self) -> Int:
        return self.size

    # ------------------------------------------------------- #
    # ---------------------- Printing ----------------------- #
    # ------------------------------------------------------- #

    fn __str__(self) -> String:
        return self._construct_output_string()

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self._construct_output_string())

    fn head(self, n: Int = 5) -> String:
        return self._construct_output_string(n)

    fn tail(self, n: Int = 5) -> String:
        return self._construct_output_string(n, True)

    # ------------------------------------------------------- #
    # ---------------------- GET / SET----------------------- #
    # ------------------------------------------------------- #

    fn get_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._ptr

    fn __getitem__(self, idx: Int) -> SeriesView[dtype]:
        return self._view(idx)

    fn __getitem__(self, slice: Slice) -> SeriesView[dtype]:
        # FIX THIS
        if slice.start and slice.end and slice.step:
            start = slice.start.value()
            end = slice.end.value() - 1
            step = slice.step.value()
        elif slice.start and slice.end:
            start = slice.start.value()
            end = slice.end.value() - 1
            step = 1
        elif slice.start:
            start = slice.start.value()
            end = size - start - 1
            step = 1
        else:
            start = 0
            end = size - 1
            step = 1

        return self._view(start, end, step)

    fn __setitem__(mut self, idx: Int, val: Scalar[dtype]):
        self._ptr[idx] = val

    # ------------------------------------------------------- #
    # ---------------------- Operations --------------------- #
    # ------------------------------------------------------- #

    # ------------------- Series & Scalars ------------------ #

    fn __add__(self, rhs: Scalar[dtype]) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            p.store[width=simd_width](i, a + rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)
        return Self(p)

    fn __iadd__(self, rhs: Scalar[dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            self._ptr.store(i, a + rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __sub__(self, rhs: Scalar[dtype]) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            p.store(i, a - rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __isub__(self, rhs: Scalar[dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            self._ptr.store(i, a - rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __mul__(self, rhs: Scalar[dtype]) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            p.store(i, a * rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __imul__(self, rhs: Scalar[dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            self._ptr.store(i, a * rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __div__(self, rhs: Scalar[dtype]) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            p.store(i, a / rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __idiv__(self, rhs: Scalar[dtype]):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            self._ptr.store(i, a / rhs)

        vectorize[closure, Self.SIMD_WIDTH](size)

    # ------------------- Series & Series ------------------- #

    fn __add__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            p.store(i, a + b)

        vectorize[closure, Self.SIMD_WIDTH](size)
        return Self(p)

    fn __iadd__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            self._ptr.store(i, a + b)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __sub__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            p.store(i, a - b)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __isub__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            self._ptr.store(i, a - b)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __mul__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            p.store(i, a * b)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __imul__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            self._ptr.store(i, a * b)

        vectorize[closure, Self.SIMD_WIDTH](size)

    fn __div__(self, rhs: Self) -> Self:
        var p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            p.store(i, a / b)

        vectorize[closure, Self.SIMD_WIDTH](size)

        return Self(p)

    fn __idiv__(self, rhs: Self):
        @parameter
        fn closure[simd_width: Int](i: Int):
            a = self._ptr.load[width=simd_width](i)
            b = rhs._ptr.load[width=simd_width](i)
            self._ptr.store(i, a / b)

        vectorize[closure, Self.SIMD_WIDTH](size)

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

    @always_inline
    fn _construct_output_string(
        self, n: Optional[Int] = None, tail: Bool = False
    ) -> String:
        var str: String
        if self.name:
            str = (
                "name='"
                + self.name.value()
                + "'"
                + " shape=("
                + String(size)
                + ",) "
                + String(dtype)
                + "\n"
            )
        else:
            str = (
                "name='' shape=(" + String(size) + ",) " + String(dtype) + "\n"
            )
        if n and tail:
            for i in range(size - n.value(), size):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str += " " * pad
                str += idx_s + " | " + String(self._ptr[i]) + "\n"
        elif n:
            for i in range(n.value()):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str += " " * pad
                str += idx_s + " | " + String(self._ptr[i]) + "\n"
        else:
            for i in range(size):
                idx_s = String(i)
                pad = Self.IDX_WIDTH - len(idx_s)
                str += " " * pad
                str += idx_s + " | " + String(self._ptr[i]) + "\n"
        return str

    fn _view(
        self, offset: Int = 0, len: Int = 0, stride: Int = 1
    ) -> SeriesView[dtype]:
        view = SeriesView(self._ptr, offset, len, stride)
        return view^

    # ------------------------------------------------------- #
    # -------------------- Constructors --------------------- #
    # ------------------------------------------------------- #

    @staticmethod
    @always_inline
    fn zeros() -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)
        memset_zero(p, size)
        return Self(p)

    @staticmethod
    fn ones() -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        for i in range(size):
            p[i] = 1
        return Self(p)

    @staticmethod
    fn randn(mu: Float64 = 0.0, sigma: Float64 = 1.0) -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)
        randn(p, size, mean=mu, standard_deviation=sigma)
        return Self(p)

    @staticmethod
    fn rand(min: Float64 = 0.0, max: Float64 = 1.0) -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)
        rand(p, size, min=min, max=max)
        return Self(p)

    @staticmethod
    fn arange(start: Int = 0, step: Float64 = 1) -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)
        @parameter
        for i in range(size):
            p[i] = Scalar[dtype](start + i * step)
        return Self(p)

    @staticmethod
    fn linspace(start: Scalar[dtype], stop: Scalar[dtype]) -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        if size == 1:
            p[0] = start
        else:
            step = (stop - start) / Scalar[dtype](size - 1)
            @parameter
            for i in range(size):
                p[i] = start + step * Scalar[dtype](i)
        return Self(p)

    @staticmethod
    fn full(value: Scalar[dtype]) -> Self:
        p = UnsafePointer[Scalar[dtype]].alloc(size)

        @parameter
        for i in range(size):
            p[i] = value
        return Self(p)

fn main() raises:
    s = Series[DType.float32, 4032].randn()
    a = Series[DType.float32, 5].ones()
    p = Series[DType.float32, 10].arange(0, 0.2)
    t = Series[DType.float32, 10].full(42)
    lin = Series[DType.float32, 10].linspace(3.2, 4.3914)
    print(lin)
