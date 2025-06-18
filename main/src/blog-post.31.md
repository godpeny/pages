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
### Convolution over volume
![alt text](images/blog31_convolution_over_volume_1.png)
You can stack multiple 2-D pixel matrix(height * width) to represent 3-D tensor(height * width * #channel). One thing to note is to match the number of channel of image and filters. So how do you convolve this pixel tensor with the 3D filter? Similar to 2-D convolution, take each numbers of the filter and multiply them with the corresponding numbers from the each channel of the tensor.  
To put it simply using above image example, take the each $9$ numbers from the $3$ channels and multiply it with the corresponding $27$ numbers that gets covered by first left yellow cube show on the image. Then add up all those numbers and this gives you this first number in the output, and then to compute the next output you take this cube and slide it over by one, and again, due to 27 multiplications, add up the 27 numbers, that gives you this next output and so on.

From the above image, you have to match the number of channel to $3$. When performing image processing in convolutional neural network, each channel represents a color. In a color image, there are three channels: red, green, and blue. An RGB image can be described as a $w \times h \times n\_c$ matrix, where each denotes the width, height, and the number of channels respectively. Thus, when an RGB image is processed, a three-dimensional tensor is applied to it.  

Unlike RGB images, grayscale images are singled channeled and can be described as a $w \times h$ matrix, in which every pixel represents information about the intensity of light.

### Multiple Filters
![alt text](images/blog31_convolution_over_volume_2.png)
When we want to detect not just single feature from the tensor, but multiple features, we can use multiple filters. (e.g., detect vertical, horizontal, 45 degree edges and so on)
The output is the stack of the result of each convolution of filter. From the above image example, since it is using two filters to detect vertical and horizontal edges, the number of output is also two and you combine these two result.
$$
\begin{aligned}
&\textbf{Input tensor:}  && n \times n \times n_c \\[2pt]
&\textbf{Filter (kernel):} && f \times f \times n_c \\[2pt]
&\textbf{Output tensor:} &&
      (\,n - f + 1\,)\;\times\;(n - f + 1)\;\times\;n_c' \\[6pt]
%-----------------------------------------------------------
&\textit{Example:}  && 6 \times 6 \times 3 
      \;\;{\xrightarrow{\;\;3 \times 3 \times 3,\; n_c'=2\;\;}}
      \;\;4 \times 4 \times 2
\end{aligned}
$$
### Convolutional Layer
![alt text](images/blog31_convolutional_layer.png)
Consider above convolutional layer example.  
$a^{[0]} = x$ is convolution with each filter $w^{[1]}$ and bias $b^{[1]}$ is added. So the result $z^{[1]} = w^{[1]} a^{[0]} + b^{[1]}$. Applying Relu function to $z^{[1]}$ to get output $a^{[1]}$.

- $f^{[l]}$ is filter(kernel) size.
- $p^{[l]}$ is padding.
- $s^{[l]}$ is stride.
- $n_c^{[l]}$ is number of channel of output(=number of filters). 
- $f^{[l]} \times f^{[l]} \times n_c^{[\,l-1]}$ is each filter.
- $f^{[l]} \times f^{[l]} \times n_c^{[\,l-1]} \times n_c^{[l]}$ is weight.
- $b^{[l]}: 1 \times 1 \times 1 \times n_c^{[l]}$ is bias.
- $a^{[l-1]} = n_H^{[\,l-1]} \times n_W^{[\,l-1]} \times n_c^{[\,l-1]}$ is input.
- $a^{[l]} = n_H^{[\,l]} \times n_W^{[\,l]} \times n_c^{[\,l]}$ is output.  

Note that the number of depth(channel) of output layer is same as the number of filters applied.
Also remind that size of height and width is below.
$$
n_H^{[\,l]} = 
\Bigl\lfloor \frac{n_H^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor,
\qquad
n_W^{[\,l]} = 
\Bigl\lfloor \frac{n_W^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor 
$$
So the output layer dimension is, 
$$
\Bigl\lfloor \frac{n_H^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor \times \Bigl\lfloor \frac{n_W^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor  \times n_c^{[l]}
$$
### Pooling
A pooling layer is a kind of network layer that downsamples and aggregates information that is distributed among many vectors into fewer vectors. So pooling is basically convolution over data with filter but summarizing the features within the region covered by the filter.
- Max Pooling: Max pooling selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map. Max pooling layer preserves the most important features (edges, textures, etc.) and provides better performance in most cases.
- Average Pooling: Average pooling computes the average of the elements present in the region of feature map covered by the filter. Thus, while max pooling gives the most prominent feature in a particular patch of the feature map, average pooling gives the average of features present in a patch. Average pooling provides a more generalized representation of the input. It is useful in the cases where preserving the overall context is important. Also it is sometimes used in very deep neural network to collapse the representation.($n \times n$ to $1 \times 1$)

The output dimension can be calculated just like before.
$$
\Bigl\lfloor \frac{n_H^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor \times \Bigl\lfloor \frac{n_W^{[\,l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigr\rfloor  \times n_c^{[l]}
$$

### Convolutional Network Example
General trend in Convolutional Network is that as go deeper in the network, size of height & width stays same for while and trend down, while the number of channer gradually increase.

![alt text](images/blog31_convolutional_neural_network_1.png)
![alt text](images/blog31_convolutional_neural_network_2.png)



### Why Convolutional Network?
- Good at detecting patterns and features in images, videos, and audio signals.
- Robust to translation, rotation, and scaling invariance.

#### Invariance
Invariance means that you can recognize an object as an object, even when its appearance varies in some way. This is generally a good thing, because it preserves the object's identity, category, (etc) across changes in the specifics of the visual input, like relative positions of the viewer/camera and the object.
![alt text](images/blog31_invariance.png)

#### Why Convolutional Network have small number of parameters?
- Parameter sharing: A feature detector (such as a vertical
edge detector) that’s useful in one part of the image is probably
useful in another part of the image.
- Sparsity of connections: In each layer, each output value
depends only on a small number of inputs.

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

