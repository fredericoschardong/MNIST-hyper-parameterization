# Programming Exercise 5: Feed Forward Single/Multiple-Hidden Layer Classifier for MNIST Dataset
## Description
Python (sklearn-based) implementation that explores how different parameters impact a feed-forward neural network with single/multiple fully-connected hidden layer(s). 

A brief analysis of the results is [provided in Portuguese](https://github.com/fredericoschardong/programming-exercise-5-MNIST-database-hyper-parameterization/blob/master/report%20in%20portuguese.pdf). It was submitted as an assignment of a graduate course named [Connectionist Artificial Intelligence](https://moodle.ufsc.br/mod/assign/view.php?id=2122514) at UFSC, Brazil.

In short, multiple normalization methods are evaluated in a single-layer FFNET for classifying handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) with multiple training algorithms, learning rate (alpha), epochs, and activation functions. Then, the best results are submitted to multiple multi-layer of fully connected perceptrons for comparison.

## Normalization

Before normalization | MinMax normalization  |  MaxAbs normalization
:-------------------------:|:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20training%20data%20before%20normalization.png "") | ![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20MinMaxScaler().png "")  |  ![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20MaxAbsScaler().png "")
L2 normalization | (x - u) / s normalization  |  Quantil-Uniform normalization
![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20Normalizer().png "") | ![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20StandardScaler().png "")  |  ![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20QuantileTransformer(output_distribution%3D'normal').png "")
Quantil-Normal normalization
![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Histogram%20after%20normalization%20with%20QuantileTransformer().png "")




## Result
Confusion matrix of the experiment with the highest f1-score (0.93) of the multi-layer experiments.

![](https://raw.githubusercontent.com/fredericoschardong/MNIST-hyper-parameterization/master/Confusion%20matrix.png "[[97  1  0  0  0  0  0  2  0  0]
 [ 0 94  1  1  0  0  0  2  0  2]
 [ 0  3 88  0  5  1  1  1  1  0]
 [ 1  0  2 88  1  2  0  0  6  0]
 [ 0  1  1  1 93  1  0  2  0  1]
 [ 0  1  0  2  0 95  0  1  0  1]
 [ 1  0  1  1  1  0 87  1  8  0]
 [ 2  1  2  1  0  1  0 92  1  0]
 [ 0  0  0  3  0  1  0  2 94  0]
 [ 0  0  1  0  0  1  0  0  1 97]]")
