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

# References

## Tensorflow
https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0

## Convolutional Neural Network
https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

## Tensor Board
https://www.tensorflow.org/get_started/summaries_and_tensorboard

### Embedding
https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py

### Scalar 
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
