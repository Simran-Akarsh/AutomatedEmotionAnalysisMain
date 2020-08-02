class CleanText(BaseEstimator, TransformerMixin):
  def remove_mentions(self, input_text):
    return re.sub(r'@\w+', ' ', input_text)

  def remove_urls(self, input_text):
    return re.sub(r'http.?://[^\s]+[\s]?', ' ', input_text)

  def emoji_oneword(self, input_text):
    #By compressing the underscore, the emoji is kept as one word
    return input_text.replace('_', ' ')

  def remove_punctuation(self, input_text):
    #Make translation table
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ') #Every punctuation symbol will be replaced by a space
    return input_text.translate(trantab)

  def remove_digits(self, input_text):
    return re.sub('\d+', ' ', input_text)

  def to_lower(self, input_text):
    return input_text.lower()
 
