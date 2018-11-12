# Modelling sequences with RNNs
As CNNs are good at dealing with grids of spatially coherent data,
RNNs are good at dealing with sequences or time-coherent data.

Again, this is an application of parameter sharing. If we had to
separate parameters for each value of the time index, we wouldn't
be able to generlize to sequence lengths not seen during training -
which is important because the relevant signal is not going to be
seen at all points in the sequence.

Convolutions allow networks to share parameters across time
but are shallow. RNNs share parameters in a different way - each
member of the output is a function of the previous members
of the output.

RNNs can be applies over multiple dimensions, you just need
a time dimension on one axis.

## Unfolding the graph
Consider $s^t = f(s^{t - 1}; \theta)$. We can unfold this
recursively: $s^t = f(f(s^{t - 2}; \theta); \theta)$;.

This can be represented as a DAG where our chain rule for
backprop just goes further and further back in time.

Generally speaking we don't actually execute all the functions. Rather
what we have is a *hidden layer* $h$ which is an approximation
of all the computation that has happened up to the point $t$. Eg, for
predicting the next word in the sentence, we only need to store the
representation of the time series which helps us best predict the next
word, not the entire series. The most demanding situation is when
we try to recover the entire input.

## RNNs generally
Generally the design pattern is as follows:
 - (1) Produce an output at each timestep and have recurrent connections
   between the hidden units
 - (2) Only have a recurrent connection from one output at one
   timestep to the hidden unit at the next.
 - (3) Read an entire sequence, then rproduce a single output.

## Teacher forcing
Models that have recurrent connections from their outputs back
into their models can be trained with teacher forcing - which
emerges from the maximum likelihood criterion. The model receives
a ground truth at time $t + 1$ - the conditional maximum
likelihood criterion is:

$\log p (y^1, y^2 | x^1, x^2) = \log p(y^2|y^1, x^1, x^2) + \log p (y^p|x^1, x^2)$

This specifies during trainign that rather than feeding the model's own
output back into itself, these connections shoudl be fed with the target values
specifying what the correct output should be. This allows us to avoid
BPTT in models that lack hidden-to-hidden connections.

## BPTT

## LSTMs
Gated RNNs are based on the idea of grading paths through time that have derivatives
that neight vanish or explode.

Leaky units allow the network to accumulate information (such as evidence
for for a particular feature) over a long diration. However, once the information
has been used, it may be useful for the network to "forget" the old state.

This is where LSTMs come in. The self-loop is conditioned on the context
rather tahn fixed. By making the weight of the self-loop gated (controlled
by another idden unit), the time scale of integration can be changed dynamically.

We have an input, input gate, forget gate, and output gate. The gates tell
us how much weight to apply to each of the following stages:

| Gate       | Weighting             |
|------------|-----------------------|
| Input gate | Input to network      |
| Forget gate| Self-loop             |
| Output gate| Output of entire unit |

The output of the cell can get shut-off by the output gate - all of the units
have a sigmoid nonlinearity, whereas the input unit can have any squashing nonlinearity.

## GRUs
The main difference between a GRU and an LSTM is that the single gating
unit simultaenously controls the forget factor and decision to update
the state unit. The reset and update gates can individually ignore
parts of the state vector, whereas the update gates act like conditional
leaky integrators that can gate any dimension.

## Gradient clipping
Stronly nonlinear function can suffer from strong gradient magnitude
shift - gradients can hit a "cliff".

A simple solution is gradient clipping - either by clipping the
gradient elementwise for by clipping the norm by doing

$$ g = \frac{gv}{||g||}, \text{if} ||g| > v $$.

It means that we avoid doing detrimental steps when the gradient
explodes and actually means we move away from numerically
unstable regions.

This however does not help to deal with vanishing gradients.


