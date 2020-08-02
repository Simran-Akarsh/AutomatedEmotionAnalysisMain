import nltk
nltk.download('punkt')
nltk.download('stopwords')
ct = CleanText()
sr_clean = ct.fit_transform(airlines_train.text)
sr_clean.samples(5)
empty_clean = sr_clean == ''
print('{} records have no words left after texting cleaning'.format(sr_clean[empty_clean].count()))
sr_clean.loc[empty_clean] = '[no_text]'


