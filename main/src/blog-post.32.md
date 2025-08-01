# Interpretability of Neural Network (Explainable artificial intelligence, XAI)
## Interpreting Neural Networks' outputs
### Saliency Maps
![alt text](images/blog32_saliency_maps.png)
In computer vision, a saliency map is an image that highlights the most relevant regions for machine learning models. The goal of a saliency map is to reflect the degree of importance of a pixel to the ML model. Simply speaking, saliency maps can be interpreted as a heat map where hotness refers to those regions of the image which have a big impact on predicting the class which the object belongs to.  

Saliency Maps can be attained by computing the gradient of the prediction (or classification score) with respect to the input features. The interpretation of the gradient attribution is that if I were to increase the color values of the pixel, the predicted class probability would go up (for positive gradient) or down (for negative gradient). The larger the absolute value of the gradient, the stronger the effect of a change of this pixel. So the corresponding pixel has an impact on the score.

One more thing to note is that the score should be pre softmax instead of post softmax. This is because in order to maximize the post score of the target, minimize the pre softmax score of other candidates.

### Occlusion Sensitivity
![alt text](images/blog32_occlusion_sensitivity.png) 
Occlusion sensitivity is a simple technique for understanding which parts of an image are most important for a deep network's classification. You can measure a network's sensitivity to occlusion in different regions of the data using small perturbations of the data.  

It is represented as a probability map of the true class for different positions of the grey square(perturbations). For example, if the input is a picture with a pomeranian, the true class is pomeranian. So if you see the occlusion sensitivity probability map of this input, the region of picture with pomeranian's face will have different color scale than other regions. This is because the color of the region(dog's face) indicates low confidence on the true class for the corresponding position of the grey square, while the color of the other region indicates high confidence on the true class
for the corresponding position of the grey square.
### Class Activation Maps
![alt text](images/blog32_class_activation_map.png) 
A Class Activation Map is for a particular category to indicate the discriminative image regions used by the
CNN to identify that category.  

The network largely consists of convolutional layers, and just before the final output layer (softmax in the case of categorization), we perform global average pooling on the convolutional feature maps and use those as features for a fully-connected layer that produces the desired output (categorical or otherwise).

Given this simple connectivity structure, we can identify
the importance of the image regions by projecting back the
weights of the output layer on to the convolutional feature
maps, a technique we call class activation mapping.

#### Mathmatic Interpretation of CAM
From above image, let $f_k(x, y)$ represent the activation
of unit $k$ in the last convolutional layer at spatial location $(x, y)$. Then, for unit $k$, the result of performing global average pooling, $F_k = \Sigma_{x,y} f_k(x, y)$. Thus, for a given class $c$, the input to the softmax, $S_c = \Sigma_{k}w^{c}_k F_k$ where $w^{c}_k$
is the weight corresponding to class $c$ for unit $k$, in other words $w^{c}_k$ indicates the importance of $F_k$ for class $c$(how much the score $S_c$ was dependent on certain feature map(unit)). Finally the output of the softmax for class $c$, $P_c$ is given by $\frac{\exp(S_c)}{\Sigma_{c} \exp(S_c)}$. (ignore the bias term)  

From the previous knowledge $F_k = \Sigma_{x,y} f_k(x, y)$ and putting into $S_c$, we get,
$$
S_c = \sum_k w_k^c \sum_{x, y} f_k(x, y) = \sum_{x, y} \sum_k w_k^c f_k(x, y)
$$

If we define $M_c$ as the class activation map for class $c$, where
each spatial element is given by,
$$
M_c(x, y) = \sum_k w_k^c f_k(x, y)
$$
Note that, $\Sigma_{x,y} M_c(x, y) = S_c$ so therefore $M_c(x, y)$ directly indicates the importance of the activation at spatial grid $(x,y)$ leading to the classification of an image to class $c$.  
Also note that from $\sum_{x, y} \sum_k w_k^c f_k(x, y)$, you can see that global average pooling doesn’t kill spatial information, unlike flattening, we know what feature maps(units) are and we can exactly map it back.

CAM expects each unit to be activated by some visual pattern within its receptive field. For example, from the above example, the blue edge unit captures the face of kid and the green edge unit caputres the face of dog. Thus $f_k$ is the map of the presence of this visual pattern.  
So the class activation map is simply a weighted linear sum of the presence of these visual patterns at different spatial locations. By simply upsampling the class activation map to the size of the input image, we can identify the image regions most relevant to the particular category.

## Visualizaing Neural Networks from the inside
### Class Model Visualization
Class Model Visualization is a technique for visualising the class models, learnt by the image classification convolutional neural network. Given a learnt classification neural network and a class of interest, the visualisation method consists in(구성하다) numerically generating an image, which is representative of the class in terms of the convolutional neural network class scoring model.  

The procedure is related to the ConvNet training procedure, where the back-propagation is used to optimise the layer weights. The difference is that in our case the optimisation is performed with respect to the input image($I$), while the weights are fixed to those found during the training stage. Also initialize the optimisation with the zero image (the ConvNet was trained on the zero-centred image data), and then added the training set mean image to the result.  

$$
L = s_{\text{dog}}(x) - \lambda \left\| I \right\|_2^2 \\[6pt]
I = I + \alpha \frac{\partial L}{\partial I}
$$
Above is the loss function and the gradient ascent method using the loss function. As mentioned before, keep the weights fixed and use gradient ascent on the input image to maximize this loss. (Remind that gradient ascent method is used to maximize the loss)  
In other word, you repeat this process below.
1. Forward propagate image $I$
2. Compute the objective $L$
3. Backpropagate to get $\frac{\partial L}{\partial I}$
4. Update $I$’s pixels with gradient ascent.

From the above expression, $\lambda$ is regularization parameter. The reason for regularization is that we don't want to have extreme value at pixel because it doesn't help. So all the values are around each other and then rescale to $0 \sim 255$ values. Since the gradient ascent doesn't constrain the input to be $0 \sim 255$, it could be $\infty$, while numbers are stored between $0 \sim 255$.

Also note that the original paper used the (unnormalised) class scores $S_c$, rather than the class posteriors returned by the soft-max layer, $P_c = \frac{\exp(S_c)}{\Sigma_{c} \exp(S_c)}$. Just like the section "Saliency Maps" with the same reason that the maximization of the class
posterior can be achieved by minimising the scores of other classes.    
The authors also experimented with optimising the posterior $P_c$, but the results were not visually prominent, thus confirming their intuition.

### Dataset Search
Dataset Search is literally a way to search what examples in the dataset lead to a strongly activated feature map given a filter. You can find out what the activation units in the middle of the network is thinking.

![alt text](images/blog32_dataset_search.png) 
For example, let's see above example. You select a certain layer whose shape is (5,5,256). In other words, there are 256 filters applied to this layer and therefore 256 (5,5) features maps are generated. Now, you choose one feature among them and see what examples in the dataset lead to a strongly activate that one feature map.  
From the image above, feature map upper case shows the images of the shirts and this indicates that the filter detects the shirts and the feature map activated the most to the image of the shirts. Similarly, the feature map lower case shows the images of the edges and this indicates that this filter detects the edges and the feature map activated the most to the image of the edges.

One question is how can we crop the input image from the activation and why is it cropped?
![alt text](images/blog32_dataset_search2.png) 
Intuition is that when you pick one activation unit as above, that activation unit doesn't see the entire image but only subpart of the image. So from the activation unit and map it back through the previous layers using the strides and filter size applied, you can get the sub part of the image that activation is seeing. Thus, the other parts of the image have no influence on that particular activation unit.  
Now let's think when you add more convolutional layers. Since you will apply more filters, naturally, one activation will cover the more part of the image. So the deeper the activation, the more it sees from the image.


### Deconvolution Network and its Application
Deconvolutional Network is a set of reversed operations of a hidden layer of a convolutional neural network. In other words, Deconvolutional Networks are convolutional neural networks (CNN) that work in a reversed process. Therefore, it has three types of layers which are the reversed max-pooling layer (unpooling layer), the reversed rectification layer(ReLU) and the reversed convolutional layer (deconvolutional layer). 

Deconvolution is also called transposed convolution, because it used transposed weight matrix that is used in convoultion. The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution. In other words, from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that
is compatible with said convolution.

![alt text](images/blog32_deconvolution.png) 
One good intuition to understand convolution and deconvolution is that convolution can be framed as mathmatical operation between matrix and vector and deconvolution is convolution with little adjustment, such as flipping weights, dividing the strides and insertion of zeros.

#### Deconvolutional Layer
Check below 4 examples and understand how deconvolution can retain original input shape of the encoding.
##### Deconvolution with no padding and no stride ($p=0, s=1$)
![alt text](images/blog32_deconvolution_no_pad_no_stride.png) 

##### Deconvolution with padding and no stride ($p \neq 0, s=1$)
![alt text](images/blog32_deconvolution_pad_no_stride.png) 

##### Deconvolution with no padding and stride ($p=0, s \neq 1$)
![alt text](images/blog32_deconvolution_no_pad_stride.png)

##### Deconvolution with padding and stride ($p \neq 0, s \neq 1$)
![alt text](images/blog32_deconvolution_pad_stride.png) 

Also note that Deconvolution allows to upsample an encoding into an image. Below exmple shows how $(4,4)$ input can be upsampled into $(6,6)$ output using deconvolution technique.

##### Deconvolution for Upsampling
![alt text](images/blog32_deconvolution_upsample.png) 
Note that each box of the output is the result of the convolution of the input with the filter. Simply speaking, the color of the box of the output is the sum of number of the same color of the input overlapped with the filter(convolution). For example, the most left upper blue box of the output is $\Sigma (255 + 134 + 123 + 94)$.

#### UnPool(Max Pool) Layer
![alt text](images/blog32_unpool.png)
Since maxpooling forgets all the numbers except for the maximum, we use cache the values. With switches(cache) you can have the exact backpropagation. For example, from the red region of the input, the other numbers ($0,1,-1$) have no impact in the loss function at the end (which means, the numbers ($0,1,-1$) Didn't pass through forward propagation).

#### UnReLU Layer
![alt text](images/blog32_unrelu.png)
Same as unpooling, When ReLU backward, we can use switch(cache) to remember which of the values in the pooing layer that had an impact on the loss. So as you can see from the above example, set 0 according to the switch and pass the rest. You pass the rest because gradient of ReLU is $1$, so just pass the gradient from the latter layer.

But in deconvolution, ReLU backward is not used but ReLU DeconvNet will be used instead. Relu DeconvNet is just you apply ReLU to output to reconstruct the input. This is hack which has been found from trial and error and not scientifically viable. The intuition behind this technique is that,
 - We are interested in which pixels of the input positively affected the activation units.
 - Also during backpropagation, we want to have minimum influence from forward propagation.

## Deep Dream
DeepDream is a computer vision program created by Google engineer Alexander Mordvintsev that uses a convolutional neural network to find and enhance patterns in images via algorithmic pareidolia (the tendency for perception to impose a meaningful interpretation on a vague stimulus, usually visual, so that one detects an object, pattern, or meaning where there is none.)

Basically, Deep Dream algorithm ses what the network is activating for and increase even this activation. Below is how dreaming process works. 
1. Forward propagate image until dreaming layer
2. Set gradients of dreaming layer to be equal to its activations
3. Backpropagate gradients to input image
4. Update Pixels of the image
5. Repeat

The reason for setting gradient of layer equal to its activations is to increase activation. Stronger the activation, the stronger it's going to become later on and so on.  

![alt text](images/blog32_deep_dream.png)
When network thought that was like tower a little bit,
you will increase the network’s confidence in the fact
that there’s a tower by changing the image and the tower comes out.

