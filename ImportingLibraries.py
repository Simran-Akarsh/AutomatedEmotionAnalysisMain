import pandas as pd
import numpy as np
from google.colab import files 
import io
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
import re

nltk.download("punkt")
nltk.download("stopwords")

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve

!pip3 install emoji
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections

import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

import gensim
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
