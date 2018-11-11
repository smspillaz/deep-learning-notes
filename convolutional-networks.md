# Convolutional Networks

These kinds of networks work best on data that has a known grid-like topology. Use
convolution in place of general matrix multiplication in at least one layer.

## Convolutions
Convolution is an operation of two functions on a real-valud argument. Example:
 - Spaceship position (t)
 - Laser sensor reading $x(t)$

We want to average several measurements with a heigher weight on more recent measurements. This
gives us a weighting functon $w(a)$. Apply a weighted average operation at every time
and we get a smoothed estimate: $s(t) = \int x(a)w(t - a) da$, or $x * w$.

For doing things like "weighted averages" and such, usually $w(a)$ must be a PDF.

Terminology: The *input* to the convolution $x$ is called the *input$ and the second argument
$w$ is called the *kernel* ("convolve $x$ by $w$"). The output is called a *feature map*.

Usually continuous convolutions, as referred to above, don't make sense in the real world,
especially when working with images. We can "revert" back to a Riemann sum instead of an
integral to get a discrete convolution: $\sum_{a = -\inf}^{\inf} x(a) w(t - a)$.

## Machine Learning
*input* is usually a multi-dimensional array, kernel usually a multidimensional array that
are *learnable parameters*. Just assume the kernel is zero except everywhere defined
so we can approximate the infinite sum using finite elements.

We also do convolution over more than one axis at a time. This is fine, just use a double integral
or a double Riemann sum. For instance we have a two-dimensional image $I$ as the input
and two dimensional kernel $K$ ($I * K$):

$$ S(i, j) = (I * K)(i, j) = \sum \sum I(m, n)K(i - m, j - n) $$

Note: Most machine learning and image processing libraries implement
a *slightly different* operator to standard convolution, called
*cross-correlation*, where the offsets $m$ and $n$ are applied to the image
as opposed to the kernel (and the kernel lives in its own space).

$$ S(i, j) = (I * K)(i, j) = \sum \sum I(i + m, i + n)K(m, n) $$

In any event, it doesn't really matter since the algorithm will just learn
the right parameters for the filters.

### Convolution as Matrix Multiplication
We can view convolution is matrix multiplication. For instance, to
do a univariate discrete convolution each row is constrained to be equal
to the row above shifted by one element, known as a Toeplitz matrix, or
a double-block circulant matrix in two dimensions. They are also
very sparse because the kernel is much smaller than the input image.

## Why Convolutions?
Three imporant ideas:
 - Sparse interactions
 - Parameter sharing
 - Equivalent representations

### Sparse Interations and Connectivity
Normal neural net layers use matrix multiplication to describe
the relationship between every input unit and every output unit.

Convolutional operators on the other hand don't use nearly as much
time if they can be implemented efficiently, meaning that we can
store fewer parameters. This is the case because of "sparse connectivity",
the phenomena that one element only receives interaction from its
nearest neighbours at the location of the convolution.

Matrix multiplication would usually require $m \times n$, but if we
limit the number of connections each output might have to $k$, then
need only do $k \times n$ operations.

Recall also the field of view across pooling layers - a neuron
in an upper layer is affected indirectly by convolutions over all
the neurons in the lower layers with a greater field of view the further
down we go.

### Parameter Sharing
A parameter can be used for more than one function in a model. In
the normal multiplication case each parameter gets used only once then
never again.

So, rather than learning a separate set of parameters for every location
we only learn one set of parameters for all locations. This is called
*equivariance to translation*.

A function is equivariant if $f(g(x)) = g(f(x))$ - in the case of
convolution let $g(x)$ be a translation operator - the convolution
is quivariant to $g$. In other words, transating and then convolving
is the same as convolving then translating.

In time series data convolution produces a timeline that shows
when different features appear in the input - so moving an event
to a later point in time doesn't change the representation, only
the time that it ocurred. For images, convolutions pick up 2D maps, so
if we move a feature in the image that feature still gets the same response
to convolution, just in its new location. Crucially, we *don't get this
with normal fully connceted layers*, which can only impose a feature
map at a static location.

## Pooling
Usually in a CNN we have three stages -> conv / nonlinearity / pooling.

Pooling replaces the output of the net at a certain location with a summary
statistics of the nearby outputs (max pooling just reports the maximum output
within a rectangular neighbourhood).

Pooling makes representations *invariant* to small translations of the input -
eg, if we translate a small amount then the values of the pooled outputs don't
change since we take the max in that region. This is useful if we want to know
*that* a feature is present rather than exactly where it is. But you wouldn't
want to use pooling in fine-grained object detection scenario (or an image
segmentation scenario) as this destroyes fine grained location information.

Generally speaking, pooling is important for handling inputs of varying
size - because the input to the classification layer must have a fixed
size.

Some other interesting research:
 - Dynamically pooling features together using clustering
 - Learning a single pooling structure

### Combining with convolution
Convolution and pooling can cause underfiting, so the priors imported
are only useful when assumptiosn made by teh prior are reasonably accurate.

So for instance, if you need to preserve precise spatial information, then
using pooling on all features can increase the training error.

## Variants of Convolution
Usually when we talk about convolution we aren't just talking about learning
one filter but about learning many filters. Usually want ot extract
lots of different features at different locations. Also convolutions are
3D -> since a color image has r/g/b and potentially even alpha channles. Finally
many implementations use 4D tensors because the fourth axis is the batch axis.

Now, because we use multichannel convolution, the operatos aren't guaranteed
to be commutative.

We have input $V_{ijk}$ (channel $i$, row/col $jk$) and $K_{ijkl}$, mapping
to channel $i$ in the output to channel $j$ in the input at an offset $k$ $l$. So
we have:

$$ Z_{i,j,k} = \sum_{l,m,n} V_{l,j+ m - 1, k + n - 1} K_{i,l,m,n} $$

We can also downsample the convolution function by applying a stride

$$ Z_{i,j,k} = \sum_{l.m,n} V_{l, (j - 1) \times s + m, (k - 1) \times s + n} K_{i,l,m,n} $$

Note: We also have to zero-pad the input otherwise the image gets smaller by one pixel
every time.

Unshared Convolution: We don't want to use convolution in some caes, but perhaps we
we just want to connect to a subset of a layer. This is sueful when we know that a feature
should be a function of a small aprt of space but is not translation invariant in general -
for instance, when detecting faces we only want to look in the bottom part of the image
for the mouth.

Tiled convolution: Whether than a separate set of weights for every spacial locations, we
have kernels taht we rotate through as we move through space. We use a different kernel for
every pixel and rotate through the knerels.

## Learning with Convolutions
Need to compute the gradient with respect to the kernel. Note that since convolution is
effectively a linear operation, we can just use standard matrix calculus to work out
the derivatives by recasting the convolution as a sparse matrix.

Here's a simple derivation:

$$ g(G, V, s)_{i, j, k, l} = \frac{\partial}{\partial K_{i, j, j, l}} J(V, K) = \sum_{m, n} G_{i,m,n} V_{j, (m - 1) \times + k, (n - 1) \times s + l} $$

If this is not the bottom of the network, then we have to compute the gradient
with respect to $V$ to backprop the error function
further down:

$$ h(K, G, S)_{i, j, k) = \frac{\partial}{\partial V_{i, j, k}} J(V, K) = \sum_{l, m} \sum_{n, p} \sum q K_{q, i, m, p} G_{q, l, n} $$

## Structure
In an non-pooled convolutional network, a filter generally produces an output which is
the proability that a given input pixel has a feature given by the feature map describing
a certain class label.

What happen if the output plane is pooled? We can emit a lower resolution grid of labels or
use a pooling operator with a unit stride. Once a prediction for each pixel is made, you can use
various methods to further process these predictions to obtain an image segmentation - we
can assume that large groups of contiguous pixels tend to be associated with the same
label.

## Types of Data in Convolutional Networks
Convolutions are great when you need to process inputs with varying spatial extents,
which doesn't really work with matrix multiplication since the dimensionality of the weights
depends on the dimensionality of the input image.

In contrast, convolutions are just applied to the image a different number of times depending
on the size of the input and the output of the convolution scales accordingly. That is, you
can convolve any $n \times m$ input with a filter of size $i \times j$. This works really
well for image segmentation, since you don't need to do anything special with overall
classification - we just need to assign a class to each pixel and we're done.

Note: Using convolution for variably sized inputs *only* makes sense when the input cells are
all describing spatially different views of the same underlying data. For instance, just
because we can convolve over a database table with different columsn that may be missing
for each record, the columns all describe independent things, so it wouldn't work. But
it would work for, eg, number of sales over time.

Examples:

| Dimensions | Single Channel          | Multichannel          |
|------------|-------------------------|-----------------------|
|1D          | Audio waveform (discrtized by time) | Skeleton animation: multiple independent waves over time |
|2D          | FFT Audio, Mono Images  | Color images          |
|3D          | Volumetric data (voxels)| Color video           |

## Efficient Convolutions
When a $d$ dimensional kernel can be expressed as the outer
product of two vectors of $d$ length (eg $d \dot d^T$), then
the kernel is called *separable*. We only need to learn the
vector representation, not the whole kernel. In fact, we
only need to do two 1-D convolutions and not a 2D convolution.

## Unsupervised Learning
The hardest part of convolutions is learning the features. What
if we could use features that were not trained by backprop?

There are some strategies for doing this. For instance, random
initialization, or designing them by hand.

Or, for instance, apply k-means to small image patches and then
use each centroid as a convolution kernel.

Random filters work surprisingly well because the fully-connected
layers can just "unscramble" the features into something sensible.

One can also learn the features using a method that doesn't require
a full backprop at every step - train the first layer in isolation,
then train the second layer and so on.
