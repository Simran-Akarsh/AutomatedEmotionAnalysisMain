cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20),columns = ['word','freq'])
fig, ax = plt.subplots(figsize=(12,10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d" , ax=ax)
plt.show();
