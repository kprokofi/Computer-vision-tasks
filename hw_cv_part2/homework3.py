import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from abc import ABC, abstractmethod

def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width

class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass

class Conv2dMatrix(ABCConv2d):
    # Функция преобразование кернела в матрицу нужного вида.
    def _unsqueeze_kernel(self, input, output_height, output_width):
        _, in_channels, in_height, in_width = input.shape
        ku_size = [self.out_channels, output_height, output_width, in_channels, in_height, in_width]
        kernel_unsqueezed = np.zeros(ku_size, dtype=np.float32)
        for i in range(output_height):
            for j in range(output_width):
                h_slice = slice(i*self.stride, i*self.stride+self.kernel_size)
                w_slice = slice(j*self.stride, j*self.stride+self.kernel_size)
                kernel_unsqueezed[:, i, j, :, h_slice, w_slice] = self.kernel
        return kernel_unsqueezed.reshape(-1, in_channels*in_height*in_width)

    def __call__(self, input_tensor):
        image_size, out_channels, output_height, output_width = calc_out_shape(
                                input_tensor.shape, 
                                self.out_channels,
                                self.kernel_size,
                                self.stride,
                                padding=0)
            
        # создадим выходной тензор, заполненный нулями         
        output_tensor = np.zeros((image_size, out_channels, output_height, output_width))
        
        # вычисление свертки с использованием циклов.
        # цикл по входным батчам(изображениям)
        for num_image, image in enumerate(input_tensor): 
             
            # цикл по фильтрам (количество фильтров совпадает с количеством выходных каналов)  
            for num_filter, filter_ in enumerate(self.kernel):
            
                # цикл по размерам выходного изображения
                for i in range(output_height):
                    for j in range(output_width): 
                        
                        # вырезаем кусочек из батча (сразу по всем входным каналам)
                        current_row = self.stride*i
                        current_column = self.stride*j
                        current_slice = image[:, current_row:current_row + self.kernel_size, current_column:current_column + self.kernel_size]
                        
                        # умножаем кусочек на фильтр
                        res = float((current_slice * filter_).sum())
                        
                        # заполняем ячейку в выходном тензоре
                        output_tensor[num_image,num_filter,i,j] = res
                        
        return output_tensor

class ABCMaxPool(ABC):
    def __init__(self, in_channels, out_channels, pool_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass

class MaxPool2D(ABCMaxPool):
    def __call__(self, input_tensor):
        image_size, out_channels, output_height, output_width = calc_out_shape(
                                input_tensor.shape, 
                                self.out_channels,
                                self.pool_size,
                                self.stride,
                                padding=0)

        mat_out = np.zeros((image_size, out_channels, output_height, output_width))

        for num_image, image in enumerate(input_tensor): 
            print(num_image, image.shape[0], image.shape)
            # цикл по фильтрам (количество фильтров совпадает с количеством выходных каналов)  
            for num_chnl in range(image.shape[0]):
                # цикл по размерам выходного изображения
                for i in range(output_height):
                    for j in range(output_width): 
                        
                        # вырезаем кусочек из батча (сразу по всем входным каналам)
                        current_row = self.stride*i
                        current_column = self.stride*j
                        current_slice = image[num_chnl, current_row:current_row + self.pool_size, current_column:current_column + self.pool_size]
                        
                        # умножаем кусочек на фильтр
                        res = float(current_slice.max())
                        print(res)
                        # заполняем ячейку в выходном тензоре
                        mat_out[num_image,num_chnl,i,j] = res
        return mat_out

class InctanceNorm:
    def __call__(self, input_tensor, gamma=1, betta=0, eps=1e-3):
        Mu = input_tensor.sum(axis=(0,1)) / input_tensor.shape[1]
        sigma = np.std(input_tensor, axis=(0,1))
        normed_tensor = ((input_tensor - Mu)/(np.sqrt(sigma**2 + eps)))*gamma + betta
        return normed_tensor

class Relu:
    def __call__(self, input_tensor):
        return np.where(input_tensor > 0, input_tensor, 0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
        
kernel = np.random.rand(5, 3, 3)

in_channels = kernel.shape[1]
out_channels = kernel.shape[0]
kernel_size = kernel.shape[2]
stride = 1
layer = Conv2dMatrix(in_channels, out_channels, kernel_size, stride)
bn = InctanceNorm()
relu = Relu()
max_pool = MaxPool2D(out_channels, out_channels, 2, 2)
layer.set_kernel(kernel)
batch_size = 1
input_height=8
input_width=8
input_matrix = np.arange(0, batch_size * in_channels *
                                input_height * input_width).reshape(batch_size, in_channels, input_height, input_width)
lay = layer(input_matrix)
pol = max_pool(lay)
print(softmax(max_pool(relu(bn(lay)))))