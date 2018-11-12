# Regularization
How do we avoid overfitting?

Regularization is any modification we make to a learning algorithm that is intended
to reduce generalization error but not training error.

There are lots of different ways of doing this. For instance, you could have
a hard constraint on parameters. Or you could have a soft constraint by penalizing
bad parameters.

Most regularization strategies are based on estimator regularization, which is trading
increased bias for reduced variance.

Brief recap: The bias-variance tradeoff is the phenomena that describes the difference
between overfitting and underfitting. You can have a model with very low variance (well
fitted) but with very high bias (assumptions which were relevant only to your sample
set, but not the population as a whole). High bias causes the algorithm to miss the more
relevant relationships between the features and targets. However, high variance means
that you are underfitted and the model can't make much sense of anything.

Models with low bias are usually more complex.

The goal of regularization is to trade a large reduction in variance for a little
bit of bias.

The main issue we run into with Deep Learning is that the algorithms are typically
applied to extremely complicated domains where generating new data would require
simulating the entire universe.

## Parameter norms
Most of the time we limit the capacity of models by adding a parameter norm penalty
to the objective function and a weight-decay parameter $\alpha$. Larger values of
$\alpha$ result in more regularization.

Note that in neural nets we only penalize the weights of the affine trnasformation
and leave the biases alone, because sometimes we need high amounts of bias to
shift the mean from the input layer to the output layer.

It is sometimes deriable to use a separate weight penalty for each layer of the network,
but this makes searching for the hyperparameters something that takes quite a long time -
so it is reasonable to just use the same weight decay at all layers to reduce the
search space size.

## L^2 regularization
Simplest penalty is the L2 norm or weight-decay. This just drives weights
closer to the origin by adding a term $\frac{1}{2} ||w||^2$ to the objective
function.

If we look closer at the gradient of the objective function, we can see that
the effect is that on each gradient step we shrink the weights by a constant
factor on each step.

The effect of this is that the component of each weight that is aligned with
the $i$-th eigenvector of the hessian of the loss function is
rescaled by a factor of $\frac{\lambda_i}{\lambda_i + \alpha}$ where $\alpha$
is the decay factor.

## L^1 regularization
This is basically just $\sum_i ||w_i||$

However, looks can be decieving. If we look at the gradient, we have:

$\triangledown_w J(w; X, y) = \alpha \text{sign}(w) + \triangledown_w J(X, y;w)$

The regularization contribution to the gradient no longer scales linearly
with each $w$, but instead it is a constant factor with a sign equal to
$\text{sign}(w_i)$.

In comparison to L2 regularization, L1 regularization results in a solution
that makes things more sparse, eg, some parameters have an optimal value of
zero. The overall result is that L1 regularization actually does feature
selection - a subset of the weights become zero meaning that their corresponding
features may be discarded.

## Under constrained problems
Regularization is necessary in order for some ML problems to become properly defined.

For instance, the solution to linear models is given by (X^TX)^{-1}. But that
doesn't work when $X^TX$ is singular, which can happen when the data-generating
distribution doesn't ahve variance in some direction or when no variance is observed
because of the fact that there are fewer examples (rows of $X$) than input
features (columsn of $X$).

Many forms of regularization resort to inversting $X^TX + \alpha I$ instead

## Data Augmentation
We can also use data as regularization. What happens if we don't have enough
data to train it on? We can create fake data and add it to the training
set as long as we have some way to do a "good enough" simulation of the
data generating process. For instance, translating, rotation, scaling
and flipping images tends ot work well in the image domain. That said there
are limites - we only have a 2D image so we can only rotate on the image
plane.

Another form of augmentation is just injecting noise, which has a regularizing
impact in the sense that it makes the network put less weight on high frequency
content as long as that noise is distributed sufficienlty randomly enough.

Dropout can also be seen as another noise-injection strategy where you essentially
multiply the layers by 0-1 noise.

## Robustness to nosie
Injecting noise can also be seen as equivalent to imposing a penalty on the
norm of the weights. This is a way of encouraging stability of
the function to be learned. It encourages the parameters to go to regions of
the parameter space where small changes in the weights have a small change
in the output - so it puhses the model into regions where the model
is relatively insensitive to small variations in the weights (eg, not points
that just happen to be minima, but minima surrounded by other flat regions).

## Noise in the outputs
For a big enough training set, the expected number of labels that would
be incorrect grows if we consider that there is some probability
$\epsilon$ that a label is incorrect. We can explicitly model
the noise on the labels - for instance that the label
$y$ is correct with probability $1 - \epsilon$, and with a probability
that it is any of the other labels $\epsilon$ otherwise.

We can incorporate this into the cost function by modelling the encoded
labels as a distrete distribution with a high peak and short tail - eg
the other labels have $\frac{\epsilon}{k - 1}$ probability and the true
label has $1 - \epsilon$.

## Semi-Supervised Learning
We have both unlabeld examples from distribution $P(x)$ and labelled examples
from $P(x, y)$ which are used to estimate $P(y | x)$ (or in otherwords,
predict the label given the data point.

This usually means learning a representation $h = f(x)$ - examples from
the same class should have similar representations, which tells
us how to group examples in representation space.

## Multitask learning
This is a way to improve generalization by pooling the examples (which
are soft constraints imposedon the parameters) arising out of several tasks.

When part of a model is shared across tasks, that part of the mdoel is more
constrained towards good values (assuming that the sharing is justified),
often yielding better generlization.

This is usually the case when we have an intermediate level representation
that both outputs can be derived from. 

## Early Stopping
We can observe both our trainign and validation error. Given enough parameters,
the model will overfit, but the validation loss will start to increase. If
we stop early, then we can obtain a model with better validation set error
by returning to the parameter setting at the point in time with the lowest validation
set error.

Every time we improve the error on the validation set we store a copy of the
weights. Return to those parameters with different hyperparameters to try
and improve the model. This can be thought of as a greedy hyperparameter search
algorithm.

Early stopping requires a vlidation set which means that we can't make use of that
data. One can perform extra trainining after the initial training once early stopping
has completed. There's two strategies for this:

 - Re-initialize the model and train again, training for the same number of iterations,
   though we don't have a good way of knowing if the number of epochs is the correct
   hyperparameter this time around given the changed data distribution
 - Keep the parameters from the first round of training but continue training with
   all the data - we don't have a guide for when to stop - we have to monitor the loss
   on the validation set and continue training until it falls below the training set
   objective. This avoids the cost but isn't as well behaved - halting problem means
   that it may never terminate.

How does early-stopping function as a regularizer? It has the effect of restricting the
optimization procedure to a relatively small volume of the parameter space in the neighbourhood
of the initial parameter value $\theta$. Limiting the number of steps limits the volume
of the parmeter space reachable from the weights.

## Parameter tying / sharing
We may want to say that certain parameter have to be close to each other.

SAy we have two models performing the same classifciation task but with somewhat different
input distributions. We believe that the parameters should be close to each other.

We can leverage this through regularization - we can use a parameter norm
penalty of $||w^A - w^B||^2.

Another mechanism is parameter sharing, usually done by sharing layers. Parameter sharing
is a thing in CNNs, eg, the cat-finding filters are going to find a cat no matter
where they are in the image since convolution is translation invariant.

## Bagging and Ensembles
This is the process of combining several models. We train the models separately
and then have all the models vote on the output for test examples (ensemble methods).

The reason why model averaging works is that different models will usually not
make all the same errors on the test set.

Now, in the case where errors are perfectly correlated, the mean squared error
doesn't reduce past the variance of the distribution. However, where the errors
are uncorrelated, then the expected squared error of the ensemble is going
to be $\frac{1}{k} E[\epsilon^2]$, in other words, the ensemble will perform at least as well
as any of its members and if the members make independent errors then the ensemble
will perform better than its members by smoothing away the errors.

### Bagging
Bagging involves constructing $k$ different datasets. Each dataset has the same number
of examples as the original dataset, but each dataset is constructed by sampling
with replacement from the original dataset. Overall, this means that each dataset has
some missing members with high probability and some duplicate members with high probability
since the probability of re-constructing the entire bag is very low.

Contents are usually won by methods of using model averaging over dozenz of models.

One technique called boosting constructs an ensemble with higher capacity than the individual
models. Boosting can be used to build ensembles of neural networks by incrementally
adding neural networks to the ensemble.

## Dropout
There are lots of different ways to think about Dropout. As described in the paper, Dropout
simulates what happens when you take the powerset of every possible model configuration
and then take the average. On expectation, we take the average of every possible subnetwork.

This is not quite the same as bagging, because in bagging all the models are independent
whereas with dropout they share parameters - they just inhit a differetn subset. This
allows us to represent an expontential number of models with a tractable amount of
memory. The process of averaging all votes form its members is just ... not applying
dropout and scaling by $p_i$ for each unit (the probability of including that unit). Eg,
we just multiply the weights by $1 - p$ at the end of trainign and then use the model
as usual.

Another view of dropout is that each hidden unit must be able to perform well regardless of
which other hidden units are in the model - eg, they must be prepared to be swapped out and
interchanged beteen the models.

## Adversarial Training
We can search for inputs that a model misclassifies by doing backprop in reverse and
injecting noise. However, we can use these same adversarial examples as a mechanism to
do adversarial training - training on adversarially perturbed examples
from the training set. This discourages highly sensitive locally linear behaviour by encouraging
the network to be locally constant in the negihbourhood of the training data.
