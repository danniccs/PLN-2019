"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -r <root> -o <file>

  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -r <root>     Input corpus root.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram


models = {
    'ngram'  : NGram,
    'addone' : AddOneNGram,
    'inter'  : InterpolatedNGram,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the corpus
    screenplay_dir = opts['-r']
    my_corpus = PlaintextCorpusReader(screenplay_dir, '.*.txt')

    sents = my_corpus.sents()

    # train the model
    n = int(opts['-n'])
    model_class = models[opts['-m']]
    model = model_class(n, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
