from collections import defaultdict
from numpy import random
from locale import strxfrm


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
                    (probs[minusgram])[token] = model.cond_prob(token,
                                                                minusgram)
                else:
                    minusgram = ()
                    (probs[minusgram])[token] = model.cond_prob(token)

        self._probs = dict(probs)

        # sort in descending order for efficient sampling

        sorted_probs = {}
        for minusgram, prob in probs.items():
            # ordeno por token y luego por la probabilidad; así se ordena por
            # probabilidad y lexográficamente

            sorted_prob = sorted(prob.items(),
                                 key=lambda pair: strxfrm(pair[0]))
            sorted_prob = sorted(sorted_prob,
                                 key=lambda pair: pair[1], reverse=True)
            sorted_probs[minusgram] = sorted_prob

        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""

        if self._n > 1:
            prev_tokens = tuple((self._n - 1) * ["<s>"])
        else:
            prev_tokens = ()

        sent = []
        cur_token = ""

        while cur_token != "</s>":
            cur_token = self.generate_token(prev_tokens)

            if cur_token != "</s>":
                sent.append(cur_token)
                if self._n > 1:
                    prev_tokens = prev_tokens[1:len(prev_tokens)]
                    aux = list(prev_tokens)
                    aux.append(cur_token)
                    prev_tokens = tuple(aux)
                else:
                    prev_tokens = ()

        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if prev_tokens is not None:
            token_probs = self._sorted_probs[prev_tokens]
        else:
            token_probs = self._sorted_probs[()]

        tokens_tuple, probs_tuple = zip(*token_probs)
        tokens = list(tokens_tuple)
        probs = list(probs_tuple)

        return random.choice(tokens, p=probs)
