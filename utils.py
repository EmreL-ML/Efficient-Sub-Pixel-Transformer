import math


def _calculate_new_conv2d_transpose_output_shape(hin=360, k_size=3, strides=2, d_rate=1, output_padding=0, padding=0):
    return hin * strides - strides - 2 * padding + d_rate * k_size - d_rate + output_padding + 1


def calculate_conv2d_transpose_same_padding(k_size=3, strides=2, d_rate=1):
    hin = 640
    hout = hin * strides
    for padding in range(10):
        for output_padding in range(10):
            if _calculate_new_conv2d_transpose_output_shape(hin=hin, k_size=k_size, strides=strides, d_rate=d_rate,
                                                            output_padding=output_padding, padding=padding) == hout:
                return padding, output_padding


def _calculate_new_conv2d_output_shape(hin=360, k_size=3, strides=2, d_rate=1, padding=0):
    return math.floor((hin + 2 * padding - d_rate * (k_size - 1) - 1) / strides + 1)


def calculate_conv2d_same_padding(k_size=3, strides=2, d_rate=1):
    hin = 2 ** 16
    hout = hin // strides
    for padding in range(10):
        if _calculate_new_conv2d_output_shape(hin=hin, k_size=k_size, strides=strides,
                                              d_rate=d_rate, padding=padding) == hout:
            return padding