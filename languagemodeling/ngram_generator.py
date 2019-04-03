from collections import defaultdict, OrderedDict
import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)

        for ngram in model._count:
            if len(ngram) == self._n:
                token = ngram[self._n - 1]
                if self._n > 1:
                    minusgram = ngram[:self._n - 1]
                    (probs[minusgram])[token] = model.cond_prob(token, nminusgram)
                else:
                    minusgram = ()
                    (probs[minusgram])[token] = model.cond_prob(token)

        self._probs = dict(probs)

        # sort in descending order for efficient sampling

        sorted_probs = defaultdict(OrderedDict)
        for minusgram, prob in probs.items():
            sorted_prob = OrderedDict(sorted(prob.items(), key=lambda pair: pair[1]))
            sorted_probs[minusgram] = sorted_prob

        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        # WORK HERE!!

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
