# Ladder Networks
Ladder networks are an unsupervised learningmethod that fits well with supervised learning.

Unsupervised learning cannot know what will be useful for the task at hand.

Paper proposes a ladder network where the auxilary task is to denoise representations at every
level of the model. Basically an autoencoder with skip connections but you can't just
rely on the skip connections, you have to denoise the input. Therefore, the intermediate
layers should provide the information necessary to do that.

## Derivation
The nice thing about hierarchical latent networks is that they can leave the details for the
lower levels to represent, which allows the higher levels to focus on more invariant, abstract
features that turn out to be relevant.

Training process can be split into inference and learning - finding the posterior
probability of the unobserved latent variables and then updating hte probability model to fit
the observations better.

Problem: How do you make the learning efficient? You have layers of latent variables $z$. We represent
the probability distribution as a product of terms and the inference process is derived from
Bayes rule.

However, there is a close connection with denoising and probabilistic modelling - given a probabilistic
model we can compute optimal denoising. Say you have $z$ and an observation $\bar z = z + n$ where $n$
is noise. First compute $p(z | \bar z)$ and then use the center of gravity as the reconstruction
$\hat z$. This minimizes the denoising cost $(\hat z - z)^2$. Given a denoising function, we can draw
samples from the corresponding distribution by creating a markov chain that alternates between
corruption and denoising.

In terms the model, we have a feedforward path that shares mappings with the corrupted feedforward
path. The decoder consists of denoising functions, trying to minimize the difference between
$\hat z$ and $z$.

The decoder works as follows:
 - Train any standard feedforward network
 - For each layer, analyze the conditional distribution of representations
   given the layer above.
 - Define a function $\hat z^l = g(\bar z^l, \hat z^{l + 1})$ which approximates
   the optimal denoising function for the family of observed distributions. The function
   $g$ is therefore expected to form a reconstruction $\hat z$ l which resembles
   the clean $z^l$ given the corruptd $\bar z^l$ and $\hat z^{l + 1}$ (eg, take the
   reconstructed data on the upper layer, combine it with noise, learn a function
   that can decode the data on the lower layer).

## How does this work with a normal perceptron?
First of all, we need to use batchnorm to prevent the denoising cost from
encouraging a trivial solution where the encoder just outputs a constant value.

Impelment corruption by adding isotropic Gaussian noise $n$ to inputs and
after each batchnorm.

The supervised learning cost is just the average NLL of the noisy output
matching the target given the inputs.

## Supporting unsupervised learning
Begin with the assumption that the noisy value of
one latent variable $\bar z$ that we want to denoise has the form
$\bar z = z + n$. We want to estimate $\hat z$ so as to minimize
$(\hat z - z)^2$

For the highest layer in the network, we choose $u = \hat y$, which allows
us to utilize information about the classes being mutually exclusive

Now, if $z$ is a truly independently distributed Gaussian set then there is
no more work for the upper layers to do. As the parameterization allows the distribution
of $z$ to be modulated by $z$ through $u$, the decoder has to find
a representation of $z$ that has high mutual information with $z^{l + 1}$, meaning
that the supervised learning has an indirect infuence on the representations learned by
the unsupervised decoder.
