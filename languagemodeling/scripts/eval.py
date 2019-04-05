"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file> -r <dir>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -r <dir>      Training corpus directory.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    screenplay_dir = opts['-r']
    my_corpus = PlaintextCorpusReader(screenplay_dir, '.*.txt')

    sents = my_corpus.sents()

    # compute the log probability, cross entropy and perplexity
    log_prob = model.log_prob(sents)
    e = model.cross_entropy(sents)
    p = model.perplexity(sents)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
