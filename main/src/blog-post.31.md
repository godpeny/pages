# Computer Vision
## Preliminaries
### Edge
Edges are defined as curves in a digital image at which the image brightness changes sharply or, more formally, has discontinuities.
These changes usually happen at borders of objects. 
Detecting edges helps us understand the shape, size and location of different parts in an image. 
### Edge Detection
Edge detection is a technique used to identify the boundaries of objects within images. 
It helps in simplifying the image data by reducing the amount of information to be processed while preserving the structural properties of the image. 
Edge Detection is essential for various image analysis tasks, including object recognition, segmentation, and image enhancement.

#### First-Order Derivative Edge Detction
In a continuous view, an edge pixel is a point where the first derivative of image intensity has a local maximum in the direction of the gradient. So the edge is where intensity changes rapidly across, and is (approximately) constant along.  
![alt text](images/blog31_intensity_function.png)
- The intensiy function is a table of samples $I[i,j]$ whose values on a pixel grid. Since the function $I$ is defined only at the lattice points $(i,j)$ like $(11, 10)$, not $(11.1, 10.5)$ (this contains nothing). So Note that intensity function is a set of discrete samples, not a continuous field.
- The gradient vector $\nabla I \;=\; \left( \frac{\partial I}{\partial x},\; \frac{\partial I}{\partial y} \right)$ points in the direction of greatest increase of intensity, where I is intensity function. So gradient magnitude $||\nabla I||$ is large at edges.  

Since the intensity function of a digital image is only known at discrete points, derivatives of this function cannot be defined unless we assume that there is an underlying differentiable intensity function that has been sampled at the image points. With some additional assumptions, the derivative of the continuous intensity function can be computed as a function on the sampled intensity function. Therefore, it turns out that the derivatives at any particular point are functions of the intensity values at virtually all image points.

However, you can get the approximate derivative of the intensity function by convolutioning filters. The discrete differentiation operator such as Sobel operator, represents a rather inaccurate approximation of the image gradient, but is still of sufficient quality to be of practical use in many applications. This operator(filter) uses convolutional masks to highlight regions with high spatial frequency, which correspond to edges.

##### Intuition
![alt text](images/blog31_vertical_image_detection.png)
Note that it doesn't matter you choose upper case to get positive output or lower case to get negative output because you take absolute value.
![alt text](images/blog31_horizontal_image_detection.png)
Note that transition region is intermediate values. These intermediate value is relatively small when image is big.

### Filter (Kernel)
In image processing, a kernel, convolution matrix, or mask is a small matrix used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between the kernel and an image.  
Put simply, during convolution, kenrel is a small grids that move over the image. Or it can be interpreted as a sliding function applied to the image matrix.  
e.g., Sobel filter.

In edge detection, instead of using existing filters, you can make the filter to learn the numbers to get good edge detector. This can be done by treating those numbers as prameters and using backpropagation.

### Convolution
Mathmatically convolution is an operation on two functions $f$ and $g$ that produces a third function $f*g$ as the integral of the product of the two functions after one is reflected about the y-axis and shifted.  
In ML, convolution is an application of a sliding window function to a matrix of pixels representing an image.

#### Convolution vs Cross-Correlation
 - Convolution in Deep Learning = Cross-Correlation
 - Convolution in Mathmatics = Cross-Correlation + Mirroring

Convolution in general, the kernel is flipped both horizontally and vertically before applying. However, in deep learning, for more computationally efficient by skipping the flipping step, the kernel is used as-is without flipping. Plus, just call convolution instead of term cross-correlation

### Padding and Stride
#### Padding
During convolution, the size of the output feature map is determined by the size of the input feature map, the size of the kernel, and the stride. If we simply apply the kernel on the input feature map, then the output feature map will be smaller than the input. This can result in the loss of information at the borders of the input feature map. In order to preserve the border information we use padding.

- Valid Convolution: No padding is added to the input feature map, and the output feature map is smaller than the input feature map. This is useful when we want to reduce the spatial dimensions of the feature maps.
- Same Convolution: Padding is added to the input feature map such that the size of the output feature map is the same as the input feature map. This is useful when we want to preserve the spatial dimensions of the feature maps.

To formalize, let's say when there is $(n \times n)$ size image and $(f \times f)$ filter. Suppose $p$ is padding size, then $(n \times n)$ image becomes $(n + 2p) \times (n + 2p)$ image after padding. After convolution, the output image is $(n + 2p - f + 1) \times (n + 2p - f + 1)$ size.
#### Stride
The number of rows and columns traversed per slide as stride. So the output size including strides is cacluated as below.
$$
\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor
\times
\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor
$$

## Convolutional Network
### Convolutional Layer
### Pooling
### Convolutional Network Example
### Why Convolutional Network?
#### Why Convolutional Network have small number of parameters?

## Convolutional Network Case Study
### Classic Networks
#### LeNet-5
#### AlexNet
#### VGG-16
### Residual Networks(ResNets)
### 1*1 Convolution
### Newwork in Network
### Inception Network

## Practical Advices for ConvNets
### Transfer Learning
### Data Augmentation
### Tips

