class CleanText(BaseEstimator, TransformerMixin):

  def remove_mentions(self, input_text):

    return re.sub(r'@\w+', '', input_text)




  def remove_urls(self, input_text):

    return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)




  def emoji_oneword(self, input_text):

    #By compressing the underscore, the emoji is kept as one word

    return input_text.replace('_', '')




  def remove_punctuation(self, input_text):

    #Make translation table

    punct = string.punctuation

    trantab = str.maketrans(punct, len(punct)*' ') #Every punctuation symbol will be replaced by a space

    return input_text.translate(trantab)




  def remove_digits(self, input_text):

    return re.sub('\d+', '', input_text)




  def to_lower(self, input_text):

    return input_text.lower()



  def remove_stopwords(self, input_text):

    stopwords_list = stopwords.words('english')

    #Some words which might indicate a certain sentiment are kept via a whitelist
    
    tokenized_text = sent_tokenize(emotions_train.content)
    print(tokenized_text)
    whitelist = []
    for line in tokenized_text:
      if line is not "'" and line is not ':' and line is not ',':
        whitelist.append(line.strip("'"))

    words = input_text.split()

    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]

    return " ".join(clean_words)




  def stemming(self, input_text):

    porter = PorterStemmer()

    words = input_text.split()

    stemmed_words = [porter.stem(word) for word in words]

    return " ".join(stemmed_words)




  def fit(self, X, y=None, **fit_params):

    return self




  def transform(self, X, **transform_params):

    clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits)

    return clean_X
