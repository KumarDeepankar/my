import pandas as pd
from pandas import DataFrame
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models, utils


# Fields in data set
# "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"


pd.options.display.float_format = '{:,.0f}'.format
regex = re.compile('[^a-zA-Z\']')
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
Trainfile = "C:\\Kaggle\\train.txt"


df = DataFrame.from_csv(Trainfile, sep='\t', header=0,index_col=None)
comments = df.iloc[:,:8]
#print(df.info())


def perform_lsi(corpus, num_topic, dictionary):

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics= num_topic)
    print(lsi.print_topics(num_topics=num_topic, num_words=20))


def perform_lda(corpus, num_topic, dictionary):
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topic)
    print(lda.print_topics(num_topics=num_topic, num_words=20))


linecount = 0
toxic_comments = []
severe_toxic_comments = []
obscene_comments = []
threat_comments = []
insult_comments = []
identity_hate_comments = []


for index, row in comments.iterrows():

    line = re.findall(r'[^"]*', row['comment_text'])
    #print(linecount)
    line = "".join(str(line).split("\\n"))
    line = regex.sub(' ', line)
    line = line.replace("'", "")
    filtered_words = ([word for word in line.lower().split() if word not in stopwords and word.isalpha() and len(word) > 1])
    lemma_words = ([lemmatizer.lemmatize(word) for word in filtered_words])

    if row['toxic'] == 1:
        toxic_comments.append(lemma_words)
    if row['severe_toxic'] == 1:
        severe_toxic_comments.append(lemma_words)
    if row['obscene'] == 1:
        obscene_comments.append(lemma_words)
    if row['threat'] == 1:
        threat_comments.append(lemma_words)
    if row['insult'] == 1:
        insult_comments.append(lemma_words)
    if row['identity_hate'] ==1:
        identity_hate_comments.append(lemma_words)


print(len(toxic_comments))
print(len(severe_toxic_comments))
print(len(obscene_comments))
print(len(threat_comments))
print(len(insult_comments))
print(len(identity_hate_comments))

toxic_dictionary = corpora.Dictionary(toxic_comments)
toxic_corpus = [toxic_dictionary.doc2bow(text) for text in toxic_comments]

severe_toxic_dictionary = corpora.Dictionary(severe_toxic_comments)
severe_toxic_corpus = [severe_toxic_dictionary.doc2bow(text) for text in severe_toxic_comments]

obscene_dictionary = corpora.Dictionary(obscene_comments)
obscene_corpus = [obscene_dictionary.doc2bow(text) for text in obscene_comments]

threat_dictionary = corpora.Dictionary(threat_comments)
threat_corpus = [threat_dictionary.doc2bow(text) for text in threat_comments]

insult_dictionary = corpora.Dictionary(insult_comments)
insult_corpus = [insult_dictionary.doc2bow(text) for text in insult_comments]

identity_hate_dictionary = corpora.Dictionary(identity_hate_comments)
identity_hate_corpus = [identity_hate_dictionary.doc2bow(text) for text in identity_hate_comments]


perform_lsi(toxic_corpus, 1, toxic_dictionary)
perform_lsi(severe_toxic_corpus, 1, severe_toxic_dictionary)
perform_lsi(obscene_corpus, 1, obscene_dictionary )
perform_lsi(threat_corpus, 1, threat_dictionary)
perform_lsi(insult_corpus, 1, insult_dictionary)
perform_lsi(identity_hate_corpus, 1, identity_hate_dictionary)

print("-------------------------------")

perform_lda(toxic_corpus, 1, toxic_dictionary)
perform_lda(severe_toxic_corpus, 1, severe_toxic_dictionary)
perform_lda(obscene_corpus, 1, obscene_dictionary )
perform_lda(threat_corpus, 1, threat_dictionary)
perform_lda(insult_corpus, 1, insult_dictionary)
perform_lda(identity_hate_corpus, 1, identity_hate_dictionary)