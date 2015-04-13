#!/usr/bin/env python
#-*-coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stopwordsEN = stopwords.words('english')
stopwordsES = stopwords.words('spanish')

tokenizer = RegexpTokenizer(r'\w+')

MIN_LENGTH=1

def extract_tags(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review)
    tokens = filter(lambda w: (w not in stopwordsEN) and (w not in stopwordsES) and len(w) > MIN_LENGTH, tokens)
    return tokens
