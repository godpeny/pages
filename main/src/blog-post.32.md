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

## Visualizaing Neural Networks from the inside
### Class Model Visualization
### Dataset Search
### Deconvolution
### Interpreting NNs using Deconvolution

### Deep Dream
