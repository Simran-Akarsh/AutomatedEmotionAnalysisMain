ct = CleanText()

sr_clean = ct.fit_transform(emotions_train.content)

sr_clean.sample(5)

empty_clean = sr_clean == ''

print('{} records have no words left after text cleaning'.format(sr_clean[empty_clean].count()))

sr_clean.loc[empty_clean] = '[no_text]'
