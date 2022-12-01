import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RDistribution():
    """
    reparameterizable distribution
    exposes a sample method which can be backpropagated through
    """
    def sample(self, temperature):
        assert False, "not implemented"

    def loss(self, other):
        assert False, "not implemented, should return something which is non negative and is zero exactly when self and other give the same sample with probability 1"
        
class DeltaDistribution(RDistribution):
    """
    Distribution that puts all of its mass on a single point
    """
    def __init__(self, value):
        self.value = value

    def sample(self, temperature):
        return self.value

    def loss(self, other):
        l1_distance = (self.value - other.value).abs()
        l1_loss = l1_distance.sum(-1).mean()
        mistakes = ((l1_distance > 1e-2)*1.).sum(-1).mean()
        return l1_loss, mistakes

class CategoricalGumbel(RDistribution):
    def __init__(self, logits):
        self.logits = logits
        
    def sample(self, temperature):
        return F.gumbel_softmax(self.logits, tau=temperature)

    def loss(self, other):
        mistakes = torch.argmax(self.logits, -1) != torch.argmax(other.logits, -1)
        mistakes = (mistakes*1.).sum(-1)

        # with probability epsilon, decided by picking category's uniformly at random
        K = self.logits.shape[-1]
        e = 1e-3
        log_random_probability = math.log(e/K**2)
        
        P = F.log_softmax(self.logits, -1)
        Q = F.log_softmax(other.logits, -1)
        log_likelihood = torch.logsumexp(P+Q, -1) + math.log(1-e)

        log_random_probability = torch.ones_like(log_likelihood)*log_random_probability

        ll = torch.stack([log_likelihood, log_random_probability], -1).logsumexp(-1).mean()

        return -ll, mistakes.mean()
