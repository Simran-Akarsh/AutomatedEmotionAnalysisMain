df_model = airlines_eda
df_model['clean_text'] = sr_clean
df_model.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(df_model.drop('airline_sentiment', axis=1), df_model.airline_sentiment, test_size=0.1, random_state=37)
mnd = MultinomialNB()
logreg = LogisticRegression()
countvect = CountVectorizer() # MultinomialNB
best_mnb_countvect = grind_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=countvect)
#joblib.dump(best_mnb_countvect, ' ../output/best_mnb_countvect.pkl')#LogisticRegression
best_logreg_countvect = grind_vect(logreg, parameters_logreg,X_train, X_test , parameters_text=parameters_vect, vect=countvect)
#joblib.dump(best_logreg_countvect, ' ../output/best_logreg_countvect.pkl')
