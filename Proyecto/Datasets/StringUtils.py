# Modulos

import pandas as pd
import re
from nltk.corpus   import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def formater(str_number):
    try:
        str_number = float(str_number)
        if str_number.is_integer():
            return "{:,.{}f}".format(str_number, 0)
        else:
            return "{:,.{}f}".format(str_number, 1)
    except:
        return str_number

def remove_accents(a):
    return list(map(unidecode,a))
  
def clean_re(s):
  def clean_temp(txt):
    txt = txt.lower()
    txt = re.sub(r'[^\w\s]',' ',txt)
    txt = ' '.join([w for w in txt.split() if len(w)>=3])
    txt = re.sub(r'\d+', ' ', txt)
    return txt
  return list(map(clean_temp,s))

def remove_stopwords(txt,stopwords):
  return list(map(lambda x :' '.join([item for item in x.lower().split() if item not in stopwords]),txt))

def get_string_stats(df,stringvar):
  df[stringvar + '_' + 'n_words'] = df[stringvar].str.split().str.len()
  df[stringvar + '_' + 'avg_len_words'] = df[stringvar].str.len() / df[stringvar + '_' + 'n_words']
  sid = SentimentIntensityAnalyzer()
  vader_=df[stringvar].apply(sid.polarity_scores)
  vader_=pd.DataFrame(map(lambda x: x,vader_))
  vader_.columns = stringvar + '_' + vader_.columns
  return (df.join(vader_))

class Vocabulary:
    def __init__(self,df,var):
        self.str_values = df[var]
        self.vocabulary = [x for y in df[var].str.split() for x in y]
        self.counter = pd.Series(self.vocabulary).value_counts()
        self.vocab_len = len(self.counter)
        self.len_sentences = df[var].str.split().str.len()
        self.longest_sentece = max(self.len_sentences)
        print(f'El tamaño de vocabulario es de {self.vocab_len} palabras únicas')
        print(f'La oración mas larga es de {self.longest_sentece} palabras')
        print(f'Las 5 palabras que mas se repiten son: {tuple(self.counter.index[0:5])}')

class VectorizeVariable:
    def __init__(self, tokenizer , X, max_features):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(X)
        self.word_index = self.tokenizer.word_index
        self.X_pad = self.tokenizer.texts_to_sequences(X)
        self.X_pad = pad_sequences(self.X_pad, maxlen=max_features)

    def transform(self,X_test):
        X_test = self.tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test,maxlen=self.max_features)
        return(X_test)

class StringCleaner():
  def __init__(self,**kwargs):
    # pass
    self.__dict__.update(kwargs)

  def clean(self,x,**kwargs):
    temp = x
    for function in list(self.__dict__.values()):
      try:
        temp = function(temp,kwargs[function.__code__.co_varnames[1]])
      except:
        temp = function(temp)
    return(temp)