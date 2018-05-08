from exceptions import SyntaxError
from collections import defaultdict
import sys
import numpy as np


def bleu(reference, candidat, n, eps=1e-5):
    ref = list(reference.split())
    cand = list(candidat.split())
    score = 0
    
    for i in range(1, n + 1):
        ref_dict = defaultdict(int)
        cand_dict = defaultdict(int)
        
        for j in range(i, len(ref) + 1):
            ref_dict[' '.join(ref[j - i:j])] += 1
        
        for j in range(i, len(cand) + 1):
            cand_dict[' '.join(cand[j - i:j])] += 1
        
        if len(cand_dict) == 0 or len(ref_dict) == 0:
            raise SyntaxError('\rError: Count of words is less than n')
        
        sum_ = 0
        for key in cand_dict.keys():
            sum_ += min(cand_dict[key], ref_dict[key])
        
        score += np.log(float(sum_) / len(cand_dict) + eps)
    
    score = np.exp(score / n)
    if len(cand) < len(ref):
        score *= np.exp(1 - float(len(ref)) / len(cand))
    return score


class Bleu:
    def __init__(self):
        self.references = {}
        self.candidates = {}
        self.n = 0
        self.score = {}
    
    def compute_score(self, references={}, candidates={}, n=4, eps=1e-5):
        #TODO: change candidates in the same type as references
        '''
        This method compute BLEU score
        
        refernces: dict, each key is <id> of block, value is the list of references strings
        candidates: dict, each key is <id> of block, value is only one candidate string
        n: int, max size of n-grams
        eps: float, constant for reason n_score == 0
        return: dict, each key is <id> of block, value is BLUE score
        
        For better understanding call method example()
        '''
        if len(references) == 0 or len(candidates) == 0:
            raise SyntaxError('\rError: compute_score() has empty input')
        
        if len(references.keys()) != len(candidates.keys()):
            raise SyntaxError('\rError: compute_score() params hame different lens')
        
        self.references = references
        self.candidates = candidates
        self.n = n
        self.score = defaultdict(int)
        
        for i in range(1, n + 1):
            ref_ngrams, average_word_count = self.__make_ref_ngrams(i)
            cand_ngrams, word_count = self.__make_cand_ngrams(i)

            for key in self.references.keys():
                ref, cand = ref_ngrams[key], cand_ngrams[key]
                n_sum = 0
                n_score = 0
                
                for ngram in cand.keys():
                    n_score += min(cand[ngram], ref[ngram])
                    n_sum += cand[ngram]
                
                if n_sum == 0:
                    raise SyntaxError('\rError: Count of words is smaller than n')
                
                if n_score != 0:
                    self.score[key] += np.log(float(n_score) / n_sum)
                else:
                    self.score[key] += np.log(eps)
                
                if i == n:
                    self.score[key] = np.exp(self.score[key] / n)
                    if word_count[key] < average_word_count[key]:
                        self.score[key] *= np.exp(1 - float(average_word_count[key]) / word_count[key])
                    
        return self.score
                
    def __make_ref_ngrams(self, n):
        assert n > 0
        ref_ngrams = {}
        average_word_count = {}
            
        for key in self.references.keys():
                
            refs = [list(block.split()) for block in self.references[key]]
            each_ref_ngrams = defaultdict(int)
            words_count = 0
                
            for ref in refs:
                ngrams = defaultdict(int)
                words_count += len(ref)
                    
                for i in range(n - 1, len(ref)):
                    ngrams[' '.join(ref[i - n + 1:i + 1])] += 1
                    
                for ngram in ngrams.keys():
                    each_ref_ngrams[ngram] = max(each_ref_ngrams[ngram], ngrams[ngram])
                
            ref_ngrams[key] = each_ref_ngrams
            average_word_count[key] = float(words_count) / len(refs)
            
        return ref_ngrams, average_word_count
        
    def __make_cand_ngrams(self, n):
        assert n > 0
        cand_ngrams = {}
        word_count = defaultdict(int)
            
        for key in self.candidates.keys():
            cand = self.candidates[key].split()
            ngrams = defaultdict(int)
            for i in range(n, len(cand) + 1):
                ngrams[' '.join(cand[i - n:i])] += 1
                
            cand_ngrams[key] = ngrams
            word_count[key] =  len(cand)
            
        return cand_ngrams, word_count
        
    def clear(self):
        self.candidates = {}
        self.references = {}
        self.n = 0
        self.score = {}
            
    def example(self):
        ref = {'1': ['the cat', 'cat'], '2': ['the dog', 'dog']}
        cand = {'1': 'cat', '2': 'dog'}
        #TODO: Add example
        pass
        
        
class CIDEr:
    def __init__(self):
        pass
    
    def add_to_corpus(self, references, n=4):
        pass
    
    def compute_score(self, candidates):
        pass
        