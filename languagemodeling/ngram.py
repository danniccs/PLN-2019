# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """
        log_sum = 0
        for sent in sents:
            log_sum += self.sent_log_prob(sent)

        return log_sum

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        # calculate log probability again so as not to go over list twice
        M = 0
        log_sum = 0

        for sent in sents:
            for token in sent:
                M += 1
            log_sum += self.sent_log_prob(sent)

        return -1*(log_sum / M)

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        return 2 ** self.cross_entropy(sents)


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:

            sent_fixed = (n-1)*['<s>'] + sent + ['</s>']

            for i in range(len(sent_fixed) - n + 1):
                ngram = tuple(sent_fixed[i:i+n])
                count[ngram] += 1

                nminusgram = tuple(sent_fixed[i:i+n-1])
                count[nminusgram] += 1

            # Leo el último n-1 grama si n>1
            if n > 1:
                nminusgram = tuple(sent_fixed[len(sent_fixed) - n + 1:
                                              len(sent_fixed)])
                count[nminusgram] += 1

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if self._n > 1:
            denominator = self.count(prev_tokens)
            numerator_list = list(prev_tokens)
        else:
            denominator = self.count(())
            numerator_list = []

        numerator_list.append(token)
        numerator = self.count(tuple(numerator_list))

        if denominator > 0:
            prob = numerator/denominator
        else:
            prob = 0

        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """

        n = self._n
        prob = 1
        fixed_sent = (n-1)*['<s>'] + sent + ['</s>']

        for i in range(n - 1, len(fixed_sent)):
            prev = tuple(fixed_sent[i - (n - 1):i])
            word = fixed_sent[i]

            if prev != ():
                prob *= self.cond_prob(word, prev)
            else:
                prob *= self.cond_prob(word)

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """

        prob = self.sent_prob(sent)
        if prob > 0:
            log_prob = math.log(prob, 2)
        else:
            log_prob = -math.inf

        return log_prob


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()

        for sent in sents:
            for token in sent:
                voc.add(token)

        voc.add("</s>")
        self._voc = voc

        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if self._n > 1:
            denominator = self.count(prev_tokens)
            numerator_list = list(prev_tokens)
        else:
            denominator = self.count(())
            numerator_list = []

        numerator_list.append(token)
        numerator = self.count(tuple(numerator_list)) + 1
        denominator += self.V()

        return numerator/denominator


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')

        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        count = defaultdict(int)

        for sent in sents:
            for k in range(0, n + 1):

                sent_fixed = (k-1)*['<s>'] + sent + ['</s>']

                for i in range(len(sent_fixed) - k + 1):
                    ngram = tuple(sent_fixed[i:i+k])
                    count[ngram] += 1

        self._count = dict(count)

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            for sent in sents:
                for token in sent:
                    voc.add(token)

            self._voc = voc

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        glob_gamma = self._gamma
        accum = 1
        prob = 0
        gamma = 0

        prev_i = prev_tokens

        for i in range(self._n, 0, -1):
            if i > 1:
                gamma = accum * self.count(prev_i)
                gamma = gamma * (1 / (self.count(prev_i) + glob_gamma))
                accum = accum - gamma
                denominator = self.count(prev_i)
                numerator_list = list(prev_i)
            else:
                gamma = accum
                denominator = self.count(())
                numerator_list = []

            numerator_list.append(token)
            numerator = self.count(tuple(numerator_list))

            if i == 1 and self._addone:
                numerator = numerator + 1
                denominator = denominator + self._V

            if prev_i is not None:
                prev_i = (prev_i)[1:len(prev_i)]

            prob = prob + gamma * (numerator / denominator)

        return prob
