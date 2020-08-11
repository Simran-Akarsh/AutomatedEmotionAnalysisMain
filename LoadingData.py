import io
from google.colab import files

train_tsv = files.upload()
airlines_train = pd.read_csv(io.BytesIO(train_tsv['Tweets.csv']))
airlines_train = airlines_train.reindex(np.random.permutation(airline_train.index))
airlines_train = airlines_train[['text', 'airlines_sentiment']]

sns.factorplot(x="airline_sentiment", data=airlines_train, kind="count", size=6, aspect=1.5, palette="GnBu_r")
plt.show();
