# Project 2 - Convolutional Neural Network for MNIST classification

## Project By : Vishal Shitole

```
Computer Engineering
North Carolina State University
Raleigh, United States
vgshitol@ncsu.edu
200258381
```

### I. STRUCTURE OF NETWORK

```
TABLE 1 - NETWORK IN TABULAR FORMAT- INITIAL NETWORK

Input Layer = Input shape (28, 28, 1)
Convolutional Layer Number of Filters = 8
Pooling Layer Pool size = (2, 2)
Convolutional Layer Number of Filters = 16
Pooling Layer Pool size = (2, 2)
Flatten
Dense 20 nodes
Dense 10 nodes
Optimizer Adam
```

### II. THE RANGE OF EACH HYPER-PARAMETERS CHOSEN

```
FOR THE EXPERIMENT

The following are the hyper-parameters chosen for tuning:
- Kernel Size
- Epochs
- Batch Size
- Learning Rate

We have done a grid-search on the parameters which have
given us the best value combination for :

- Kernel Size - (3, 3)
- Epochs - 15
- Batch Size - 128
- Learning Rate - 0.001

We have varied the all the parameters individually within
a range and during this time all the other hyper-parameters
have been kept to their best value that has been mentioned
here. Activation function has been chosen as default i.e.
Linear for tuning the hyperparameter and initial tuning. We
will later apply the model with different activation functions
and observe the performance.

RANGE OF PARAMETERS :

- Kernel Size: [ 2, 3, 4 ]
- Epochs: [ 5, 10, 15, 20 ]
- Batch Size: [ 16, 32, 64, 128, 256, 512, 1024 ]
- Learning Rate: [ 0.0001, 0.0005, 0.001, 0.002]
After tuning it on the above hyper-parameters, we have
tested the varied the network with 3 activation functionstanh,
reluandsigmoid. We have chosen the best activation function
to test on our validation set.

```
### III. LEARNING PARAMS FOR TUNING DNN MODEL
```
There may seem some discrepancies between the best value
in the graphs and the best value given in by the grid-search. We
have trained on randomly chosen training sets. Since we have
done the training separately on 2 models this discrepancy is
showing. That is, the models for training we not saved for both
best value graph and grid-search, so there is bit of mismatch
in the graphs and grid-search.

We have varied the activation functions based on the best
values on the grid-search. The activation functions considered
for the graphs are tanh, sigmoid, relu.

We find the best accuracy is obtained for tanh activation.

Final structure of the network

Epoch: 15
Batch Size: 128

TABLE II : NETWORK INTABULAR FORMAT-AFTER HYPER-PARAMETER TUNING

Input Layer Data from MNIST digits set
Convolutional Layer Input shape = (28, 28, 1) Activation: tanh, Kernel Size: 3, Number of Filters: 8
Pooling Layer Pool size = (2, 2)
Convolutional Layer Activation: tanh, Kernel Size: 3, Number of Filters: 16
Pooling Layer Pool size = (2, 2)
Flatten
Dense 20 nodes Activation: tanh
Dense 10 nodes Activation: softmax
Optimizer Adam Learning Rate: 0.
```
### IV. ANALYSIS AND DISCUSSION
```
From the figures for activation versus loss and accuracy
(Fig. 6 to Fig. 11), the loss decreases the fastest forrelu,
while sigmoid is slowest to converge. This is evident from
the respective shape of the function and also from the graphs.
Almost, similar accuracy is achieved by all three activation
functions. We could have selectedreluas our final activation
function to decrease the time for training. However, as the
training time is not large, we resort to the activation that gives
the best accuracy here i.e.tanh
We selected only some of the many hyper-parameters that can
be tuned. But we did select the hyper parameters that have
major impact, like batch size and epochs and learning rate.
Specifically, grid search with cross validation (2-fold) was
applied as it is better to test the combinations of the hyper-
parameters, rather than fixing one and tuning the other.
```

