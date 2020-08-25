# parameter grind setting for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features_pipe_vect_max_df': (0.25, 0.5, 0.75),
    'features_pipe_vect_ngram_range': ((1,1), (1,2)),
    'features_pipe_vect_min_df': (1,2)
}
# parameter grind setting for MultionomialNB
parameters_mnb = {
  'clf_alpha': (0.25, 0.5, 0.75)
}
# parameter grind setting for LogisticRegression
parameters_logreg = {
    'clf_c': (0.25, 0.5, 1.0),
    'clf_penalty': ('l1', 'l2')
}
