# tiny_ml
# Model Optimazation
After obtaining the optimal model through tensorflow, the model needs to be optimized. Prunning and quantization techniques will be used to optimize our model.

## Prunning 
Deep neural networks (DNNs) use an immense number of parameters and therefore require powerful computer hardware during operation, and even more so for the initial training of the network. As a consequence, they become impractical when only limited hardware resources and/or time are available. Furthermore, with a large number of weights and nodes there is an increasing risk of overfitting.In addition to the neural network’s parameters, many other components and hyperparameters are to be selected and fine-tuned when designing and training neural networks, e.g., the type of layers, the batch size, the amount of dropout or the regularization and the learning rate. This pre-training processes also account for a large part of extra costs of training neural networks, which increases with its number. Neural network training thus has to compromise between at least two conflicting goals: On one hand, the prediction accuracy should be as high as possible (which usually asks for many network parameters and sophisticated hyperparameter settings), while on the other hand, the network complexity should be as low as possible and the training should require a minimum number of parameters. Network pruning methods are an effective approach to limit overparameterization of DNNs and to reduce the complexity of the network. Pruning removes edges (weights), nodes (neurons) or even feature maps (filters) from the network according to their importance. Removing parts of the network reduces the required storage capacity and speeds up the inferece time of the network (response time). 

Some kinds of pruning methods are: 

- Global Magnitude Pruning - prunes a fraction f of all the weights with the lowest absolute value.

- Layerwise Magnitude Pruning - for each layer, prunes a fraction f of the weights with the lowest absolute value.

- Global Gradient Magnitude Pruning - prunes a fraction f of all the weights with the lowest absolute value of weight times gradient, evaluated on a batch of inputs.

- Layerwise Gradient Magnitude Pruning - for each layer, prunes a fraction f of the weights with the lowest absolute value of weight times gradient, evaluated on a batch of inputs.

- Random - prunes a random fraction f of all the weights.

Magnitude-based pruning approaches are common baselines in the literature that have proven to be competitive with more complex methods. This is also the approach of tensorflow, and also the benchmark of this project.

Gradient-based methods are less common, but have recently gained popularity.

Random pruning is just a straw man approach.

There are important factors that should be carefully considered:

1. Remove weights or neurons?

- You can prune weights. This is done by setting individual parameters to zero and making the network sparse. This would lower the number of parameters in the model while keeping the architecture the same. 
Weight-based pruning is more popular as it is easier to do without hurting the performance of the network. However, it requires sparse computations to be effective. This requires hardware support and a certain amount of sparsity to be efficient. 

- You can remove entire nodes from the network. This would make the network architecture itself smaller, while aiming to keep the accuracy of the initial larger network.

2. What to prune?
 The goal is to remove more of the less important parameters.

3. When to prune?
If you are using a weight magnitude-based pruning approach, as described in the previous section, you would want to prune after training.

4. How to evaluate pruning? There multiple metrics to consider when evaluating a pruning method: accuracy, size, and computation time. 

###  Pruning in Keras 
The TensorFlow Model Optimization Toolkit is a suite of tools for optimizing ML models for deployment and execution. Magnitude-based weight pruning  is applied in Keras. 
This is the simplest weight pruning algorithm. After each training, the link with the smallest weight is removed. Thus the saliency of a link is just the absolute size of its weight. Though this method is very simple, it rarely yields worse results than the more sophisticated algorithms.

Magnitude-based weight pruning gradually zeroes out model weights during the training process to achieve model sparsity. Sparse models are easier to compress, and we can skip the zeroes during inference for latency improvements.

- Tensorflow_pruning: https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/g3doc/guide/pruning
- WHAT IS THE STATE OF NEURAL NETWORK PRUNING?: https://arxiv.org/pdf/2003.03033.pdf
- NoiseOut: A Simple Way to Prune Neural Networks: https://arxiv.org/pdf/1611.06211.pdf
- PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE: https://arxiv.org/pdf/1611.06440.pdf
- Standardizing Evaluation of Neural Network Pruning: https://shrinkbench.github.io/jjgo-aisystems2019.pdf


## Quantization: 

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. A quantized model executes some or all of the operations on tensors with integers rather than floating point values. This allows for a more compact model representation and the use of high performance vectorized operations on many hardware platforms. This technique is in particular useful at the inference time since it saves a lot of inference computation cost without sacrificing too much inference accuracies.
Quantization maps a floating point value x  in floating point interval to b-bit integer with integer interval. From this description, our quantization function as well as de-quantization function can be definded. 


There are 3 forms of quantization: post-training quantization( or static quantization), quantization aware training and dynamic quantization. Usually the static quantization and the quantization aware training are the most common to see in practice, since they are the fastest among all the three modes in practice.

1. Start with post-training quantization ( or static quantization) since it's easier to use, though quantization aware training is often better for model accuracy.

Post-training quantization is a technique in which the neural network is entirely trained using floating-point computation and then gets quantized afterward.

To do this, once the training is over, the neural network is frozen, meaning its parameters can no longer be updated, and then parameters get quantized. The quantized model is what ultimately gets deployed and used to perform inference without any changes to the post-training parameters.

Though this method is simple, it can lead to higher accuracy loss because all of the quantization-related errors occur after training is completed, and thus cannot be compensated for.

Static Quantization (Post Training Quantization) is typically used when both memory bandwidth and compute savings are important. CNNs is a typical use case.

2. Quantization aware training emulates inference-time quantization, creating a model that downstream tools will use to produce actually quantized models. The quantized models use lower-precision (e.g. 8-bit instead of 32-bit float), leading to benefits during deployment.

Quantization aware training works to compensate for the quantization-related errors by training the neural network using the quantized version in the forward pass during training.
The idea is that the quantization-related errors will accumulate in the total loss of the model during training, and the training optimizer will work to adjust parameters accordingly and reduce error overall.

Quantization-aware training has the benefit of much lower loss than post-training quantization. 

3. In dynamic quantization the weights are quantized ahead of time but the activations are dynamically quantized during inference (on the fly). Hence, dynamic.

As mentioned above dynamic quantization have the run-time overhead of quantizing activations on the fly. So, this is beneficial for situations where the model execution time is dominated by memory bandwidth than compute (where the overhead will be added). This is true for LSTM and Transformer type models with small batch size.


For TinyML, quantization is an invaluable tool that is at the heart of the whole field.

All in all, quantization is necessary for three main reasons: 

1. Quantization significantly reduces model size—this makes it more feasible to run ML on a memory-constrained device like a microcontroller.
2. Quantization allows for ML models to run while requiring less processing capabilities—MCUs used in TinyML tend to have less performant processing units than a standard CPU or GPU. 
3. Quantization allows for a reduction in power consumption—the original goal of TinyML was to perform ML tasks at a power budget under 1mW. This is necessary to deploy ML on devices powered by small batteries like a coin cell. 

So far, major deep learning frameworks, such as TensorFlow and PyTorch, have supported quantization natively. 
###  Quantization in Keras
Keras supports post-training quantization and quantization aware training, which is just the static quantization we talked about. 


- The mathematics of quantization for neural networks is explained: https://leimao.github.io/article/Neural-Networks-Quantization/
- A Survey of Quantization Methods for Efficient
Neural Network Inference: https://arxiv.org/pdf/2103.13630.pdf
- Tensorflow_quantization: https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization
