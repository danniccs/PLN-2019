import pickle
from nltk.corpus import PlaintextCorpusReader

def loadcorp():
    with open('tst.txt', 'rb') as f:
        model = pickle.load(f)
    
    corpus = PlaintextCorpusReader('../ScreenplayTXTs', 'The\ Two\ Towers.txt')
    
    return (model, corpus.sents()[19]) 
