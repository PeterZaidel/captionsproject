from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from autocorrect import spell
import numpy as np
from nltk.stem import PorterStemmer
class cider:
    def __init__(self, n_gram):
        self.vectorizer = TfidfVectorizer(ngram_range=(n_gram,n_gram))
        self.stemmer = PorterStemmer()
        
    def correct(self, text):
        symbs_to_replace = ['.', ',', '/', '-', ':', '{', '}', '[', ']', ]
        for smb in symbs_to_replace:
            text = text.replace(smb, ' ')
        words = [self.stemmer.stem(word) for word in word_tokenize(text.lower())]
        for idx in range(len(words)):
            words[idx] = spell(words[idx])
        return ' '.join(words)
    
    def correct_references(self, texts):
        return [self.correct(text) for text in texts]
    
    def fit(self, X):
        list_of_sents = [sent for descriptions in X for sent in self.correct_references(descriptions)]
        self.vectorizer.fit(list_of_sents)
        return self
    
    def fit_transform(self, X):
        list_of_sents = [sent for descriptions in X for sent in self.correct_references(descriptions)]
        return self.vectorizer.fit_transform(list_of_sents)
    
    def transform(self, X):
        list_of_sents = [sent for descriptions in X for sent in self.correct_references(descriptions)]
        print(list_of_sents)
        return self.vectorizer.transform(list_of_sents)
    
    
    # Правило - n-граммы референсов должны быть в vectorizer
    # List of refs - list of str, candidate - str
    def distance(self,list_of_refs, candidate):
        print(self.vectorizer.vocabulary_)
        tfidf_refs = self.transform([list_of_refs])
        tfidf_cand = self.transform([[candidate]]) # Убрать скобки, если кандидат будет в виде листа
        scores = np.zeros([len(list_of_refs)])
        norm_refs = np.zeros([len(list_of_refs)])
        norm_cand = tfidf_cand*tfidf_cand.T
        if norm_cand.data > 0:
            norm_cand = norm_cand.data
        else:
            return np.zeros([len(list_of_refs)])
        for i in range(tfidf_refs.shape[0]):
            score = tfidf_refs[i,:] * tfidf_cand.T
            norm_ref = tfidf_refs[i,:] * tfidf_refs[i,:].T
            norm_refs[i] = norm_ref.data if norm_ref.data > 0 else 0
            scores[i] = score.data if score.data>0 else 0
        
        result = (scores / norm_refs / norm_cand).mean()
        return result