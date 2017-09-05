# Installation

`pip3 install jupyter`

`jupyter notebook`

Copy Auth token to run in other environments


## Tensorflow POI

### Initializer
* random_normal(_initializer)
    * The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    * Import: tf.truncated_normal
* truncated_normal(_initializer)
    * The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    * Import: tf.truncated_normal

* xavier_initializer
    * This initializer is designed to keep the scale of the gradients roughly the same in all layers. In uniform distribution this ends up being the range: x = sqrt(6. / (in + out)); [-x, x] and for normal distribution a standard deviation of sqrt(2. / (in + out)) is used.
    * Import: tf.contrib.layers.xavier_initializer
    * Docs: [Xavier Initializer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
* **Examples**:
    ```
    tf.Variable(
        tf.truncated_normal([hidden_layer, output], stddev=0.1),
        name="Weights"
    )
    ```
    ```
    tf.get_variable(
        "Weights",
        shape=[hidden_layer, output],
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    ```
    ```
    tf.get_variable(
        "Weights",
        shape=[hidden_layer, output],
        initializer=tf.contrib.layers.xavier_initializer()
    )```

### Activation
* sigmoid (Classification)
  * Use simple sigmoid only if your output admits multiple "true" answers, for instance, a network that checks for the presence of various objects in an image. In other words, the output is not a probability distribution (does not need to sum to 1).
* relu (Linear Regression)
  * The best function for hidden layers

### Convolution Padding
Approaches for running weighted filters over image channels.

* Assume input image is: `4 x 4` with a grey channel ` x 1`
    * height_1 = 4
    * width_1 = 4
    * input_x_shape = `[?, 4, 4, 1]`
* Assume layer has a weighted filter of `2 x 2` with an output channel of ` x 2`
    * filter_height = 2
    * filter_width = 2
    * weight_1_shape = `[2, 2, 1, 2]` (1 = greyscale)
* Assume filter strides will be 1 in each direction `[1, 1, 1, 1]`
    * stride_height = 1
    * stride_width = 1
    * stride_shape = `[1, 1, 1, 1]`
* padding = "SAME" (The filter goes outside the boundaries of the image)
    * tf.nn.conv2d(input_x, weight_1, strides=stride_shape, padding="SAME")
    * height_2 = height_1 / stride_height
    * width_2 = width_1 / stride_width
    * preactivate_shape = height_2, width_2 output_channel
        * [4, 4, 2]

* padding = "VALID" (The filter is limited to depressions of the image)
    * tf.nn.conv2d(input_x, weight_1, strides=stride_shape, padding="VALID")
    * height_2 = (height_1 - filter_height + 1) / stride_height
        * (4 - 3 + 1 / 1) = 3
    * width_2 = width_1 / stride_width
        * (4 - 3 + 1 / 1) = 3
    * preactivate_shape = height_2, width_2 output_channel
        * [3, 3, 2]

### Max Pool
This process reduced the amount of filters that look for features by location.
If any layers have active features regardless of location max pooling squashes the results.

Ksize reduces the conv dimensions by (conv2d - pool_shape + 1) / strides

Example:
* conv2d = [1, 8, 8, 2]
* pool_shape = [1, 8, 1, 1]
* stride_pool_shape = [1, 1, 4, 1]

Result:
* `(1-1+1)/1`,
* `(8-8+1)/1`,
* `(8-1+1)/4`,
* `(2-1+1)/1`

[1, 1, 2, 2]

# References

## Tensorflow
https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0

## Convolutional Neural Network
https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

### Padding
http://deeplearning.net/software/theano/_images/numerical_padding_strides.gif

## Tensor Board
https://www.tensorflow.org/get_started/summaries_and_tensorboard

### Embedding
https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py

### Scalar
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
