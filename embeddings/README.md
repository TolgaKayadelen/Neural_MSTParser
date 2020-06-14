####   TURKISH WORD2VEC MODEL

- This is a pretrained word2vec model for Turkish where words are represented as 300 dimensional dense vectors. 
- We use skip-gram method with negative sampling, where # negative samples per word are set as 20 during training. 
- The model is trained using the following datasets:
  - The BOUN Corpus (both Generic and News subsets)
  - Wikipedia paragraphs obtained from TSCorpus (a total of ~149.200 paragraphs)
  - The sentences from IMST-UD Turkish Treebank. 
- No normalization is applied to the data during training except for cleaning of punctuation. Capital words are kept as is.
- The total number of tokens used in training the model is ~450M.


##### USAGE NOTES

- Make sure you work with Python 3.6+.

- Load the model
```
>>> import gensim
>>> model =  gensim.models.Word2Vec.load("tr-word2vec-model_v3.bin")
```

- Check a word's vector values
```
>>> word = "kelime"
>>> model[word]
```

- Try similarity checks
```
>>> word = "kelime"
>>> model.most_similar(word)
```




