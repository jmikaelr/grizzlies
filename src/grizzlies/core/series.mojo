struct Series[
        dtype: DType, size: Int, _name: Optional[String] = None
](
    AnyType & Writable & Stringable
):
    """
    A 1D labeled array holding data of a single type. The core data column.
    """

    alias IDX_WIDTH: Int = Self._digits(size)
    var _data: UnsafePointer[Scalar[dtype]]

    @always_inline("nodebug")
    fn __init__(out self):
        self._data = UnsafePointer[Scalar[dtype]].alloc(size)

    @always_inline("nodebug")
    fn __init__(out self, data: UnsafePointer[Scalar[dtype]]):
        self._data = data

    fn __del__(deinit self):
        self._data.free()

    fn __str__(self) -> String:
        var str_data: String
        if self._name:
            str_data = "name='" + self._name.value() + "'" +
            " shape=(" + String(size) + ",) " + String(dtype) + "\n"
        else:
            str_data = "name='' shape=(" + String(size) + ",) " + String(dtype) + "\n"
        for i in range(size):
            idx_s = String(i)
            pad = Self.IDX_WIDTH - len(idx_s)
            str_data += " " * pad
            str_data += (idx_s + " | " + String(self._data[i]) + "\n")

        return str_data

    fn write_to(self, mut writer: Some[Writer]):
        # if self._name:
        #     writer.write("name='" + self._name.value() + "'" +
        #     " shape=(" + String(size) + ",) " + String(dtype) + "\n")
        # else:
        #     writer.write("name='' shape=(" + String(size) + ",) " + String(dtype) + "\n")
        # for i in range(size):
        #     idx_s = String(i)
        #     pad = Self.IDX_WIDTH - len(idx_s)
        #     for _ in range(pad):
        #         writer.write(" ")
        #     writer.write((idx_s + " | " + String(self._data[i]) + "\n"))
        writer.write(self.__str__())


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



fn main():
    var s = Series[DType.float32, 6000]()
    print(s)
