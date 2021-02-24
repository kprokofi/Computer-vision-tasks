import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from abc import ABC, abstractmethod
from scipy.special import softmax

def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    ''' calculating output shape after conv layer '''
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
    def __call__(self, input_tensor):
        image_size, out_channels, output_height, output_width = calc_out_shape(
                                input_tensor.shape, 
                                self.out_channels,
                                self.kernel_size,
                                self.stride,
                                padding=0)
            
        # create output tensor filled by zeros         
        output_tensor = np.zeros((image_size, out_channels, output_height, output_width))
        
        # calculating convolution using loops
        # loop through input batches
        for num_image, image in enumerate(input_tensor): 
             
            # filter loop (num of filters equals to output channels)  
            for num_filter, filter_ in enumerate(self.kernel):
            
                # Spatiality loop
                for i in range(output_height):
                    for j in range(output_width): 
                        
                        # cut slice of the batch (over all output channels)
                        current_row = self.stride*i
                        current_column = self.stride*j
                        current_slice = image[:, current_row:current_row + self.kernel_size, current_column:current_column + self.kernel_size]
                        
                        # apply convolution transform
                        res = float((current_slice * filter_).sum())
                        
                        # fill corresponding output cell
                        output_tensor[num_image,num_filter,i,j] = res
                        
        return output_tensor


class MaxPool2D(ABCConv2d):
    def __call__(self, input_tensor):
        image_size, out_channels, output_height, output_width = calc_out_shape(
                                input_tensor.shape, 
                                self.out_channels,
                                self.kernel_size,
                                self.stride,
                                padding=0)

        mat_out = np.zeros((image_size, out_channels, output_height, output_width))

        for num_image, image in enumerate(input_tensor):   
            for num_chnl in range(image.shape[0]):

                for i in range(output_height):
                    for j in range(output_width): 
                        
                        # cut slice over one channel
                        current_row = self.stride*i
                        current_column = self.stride*j
                        current_slice = image[num_chnl, current_row:current_row + self.kernel_size, current_column:current_column + self.kernel_size]
                        
                        # do max pool 
                        res = float(current_slice.max())
                        # fill corresponding output cell
                        mat_out[num_image,num_chnl,i,j] = res
        return mat_out


class LayerNorm:
    def __call__(self, input_tensor, gamma=1, betta=0, eps=1e-3):
        result = np.zeros(input_tensor.shape)
        for b, batch in enumerate(input_tensor):
            for c, image in enumerate(batch): 
                Mu = np.mean(image)
                sigma = np.std(image)
                result[b,c,:,:] = ((image - Mu)/(np.sqrt(sigma**2 + eps)))*gamma + betta
        return result


class Relu:
    def __call__(self, input_tensor):
        return np.where(input_tensor > 0, input_tensor, 0)


def custom_softmax(x):
    # """Compute softmax values per pixel in layers."""
    e_x = np.exp(x - np.max(x, axis=1))
    return e_x / np.sum(e_x, axis=1)


class FeedForward:
    def __init__(self, stride=1):
        kernel = np.random.randint(-10, 10, 5*3*3).reshape(5,3,3)
        in_channels = kernel.shape[1]
        out_channels = kernel.shape[0]
        kernel_size = kernel.shape[2]

        self.conv2d = Conv2dMatrix(in_channels, out_channels, kernel_size, stride)
        self.conv2d.set_kernel(kernel)
        self.LN = LayerNorm()
        self.relu = Relu()
        self.max_pool = MaxPool2D(out_channels, out_channels, 2, 2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.LN(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return custom_softmax(x)


def main():
    input_img = cv.imread("..\image_example\cats.jpg")
    input_img = cv.resize(input_img, (32,32)).reshape(1, 3, 32, 32) # BxCxWxH
    Net = FeedForward(stride=1)
    out = Net.forward(input_img)
    print("forward out: \n", out)
    print("output shape: \n", out.shape)

if __name__ == '__main__':
    main()
