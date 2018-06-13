# coding: utf-8
# https://ckmarkoh.github.io/blog/2016/07/10/nlp-vector-space-semantics/ 

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy import spatial

text = [
    ["the", "dog", "run", ],
    ["a", "cat", "run", ],
    ["a", "dog", "sleep", ],
    ["the", "cat", "sleep", ],
    ["a", "dog", "bark", ],
    ["the", "cat", "meows", ],
    ["the", "bird", "fly", ],
    ["a", "bird", "sleep", ],
]


def build_word_vector(text):
    word2id = {w: i for i, w in enumerate(sorted(list(set(reduce(lambda a, b: a + b, text)))))}
    id2word = {x[1]: x[0] for x in word2id.items()}
    wvectors = np.zeros((len(word2id), len(word2id)))
    for sentence in text:
        for word1, word2 in zip(sentence[:-1], sentence[1:]):
            id1, id2 = word2id[word1], word2id[word2]
            wvectors[id1, id2] += 1
            wvectors[id2, id1] += 1
    return wvectors, word2id, id2word

#https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#Note that spatial.distance.cosine computes the distance, and not the similarity. So, you must subtract the value from 1 to get the similarity.
def cosine_sim(v1, v2):
  print("v1.v2",np.dot(v1, v2))
  print("np.sqrt(np.sum(np.power(v1, 2)) ",np.sqrt(np.sum(np.power(v1, 2))))
  print("v1= ",v1)
  print("v2= ",v2)
  print("spatial.distance.cosine(v1, v2) ",spatial.distance.cosine(v1, v2))
  
  #return np.dot(v1, v2) / (np.sqrt(np.sum(np.power(v1, 2))) * np.sqrt(np.sum(np.power(v2, 2))))
  
  result = 1 - spatial.distance.cosine(v1, v2)
  return result
  

def visualize(wvectors, id2word):
    np.random.seed(10)
    fig = plt.figure()
    U, sigma, Vh = np.linalg.svd(wvectors)
    ax = fig.add_subplot(111)
    ax.axis([-1, 1, -1, 1])
    for i in id2word:
        ax.text(U[i, 0], U[i, 1], id2word[i], alpha=0.3, fontsize=20)
    plt.show()


_wvec,_word2id,_id2word=build_word_vector(text)
# visualize(_wvec,_id2word)

_similarity=cosine_sim(_wvec[_word2id["the"]],_wvec[_word2id["a"]]) 
print("Similarity = ",_similarity)

