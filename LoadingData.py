emotions_csv = files.upload()
emotions_train = pd.read_csv(io.BytesIO(emotions_csv['text_emotion.csv']))
emotions_train = emotions_train.reindex(np.random.permutation(emotions_train.index))
emotions_train = emotions_train[['content','sentiment']]
emotion_file = files.upload()
emotion_file = open('emotions.txt',encoding='utf-8').read()

sns.factorplot(x="sentiment", data=emotions_train, kind="count", size=10, aspect=1.5, palette="GnBu_r")
plt.show();
