#Transformers
#Pretrained Transformer

from sentence_transformers import SentenceTransformer, util
start_time = time.time()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
corpus_embeddings = model.encode(df.sn_short_description.values, convert_to_tensor=True)
#query_embedding = model.encode(query_sentence, convert_to_tensor=True)


# code to be executedimport torch
import time

query_embedding = model.encode(query_sentence, convert_to_tensor=True)
# We use cosine-similarity and torch.topk to find the highest 3 scores
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=10)
query_sentence = ' Png failed for SAWP2580091.gbl.ad.hedani.net from PingMonitoring'
print("\n\n======================\n\n")
print("Query:", query_sentence)


for score, idx in zip(top_results[0], top_results[1]):
    score = score.cpu().data.numpy() 
    idx = idx.cpu().data.numpy()
    
display(df[['sn_short_description']].iloc[best_index])    
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))





















import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_csv("incident_data.csv")

X = df['incident number']
y = df['KD Link']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert the training data into Tf-idf vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Convert the testing data into Tf-idf vectors
X_test_vectors = vectorizer.transform(X_test)

# Train the model using the training data
model = SomeSimilarityModel()
model.fit(X_train_vectors, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_vectors)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)










df = df[~df.sn_short_description.isna()]
df = df.iloc[:20000]
df.head()
STOPWORDS = set(stopwords.words('english')).union({'esc211', 'esc', 'crit', 'warn', 'val', 'percent', 'pct', 'severity', 'warning', 'rule', 'initial', 'alert','l','esc2', 'sgw','gbw', 'usw', 'uslp', 'krlp', 'max', 'min', 'val','cs', 'ap','prod', 'lin'})
MIN_WORDS = 4
MAX_WORDS = 200




def clean_text(text):
    """
    Series of cleaning. String to lower case, remove non words characters and numbers.
        text (str): input text
    return (str): modified initial text
    """
    PATTERN_S = re.compile("\'s")  # matches `'s` from text`
    PATTERN_RN = re.compile("\\r\\n") #matches `\r` and `\n`
    PATTERN_PUNC = re.compile(r"[^\w\s]") # matches all non 0-9 A-z whitespace
    PATTERN_1 = re.compile(r"\b(?<!-)[0-9]+\b\s*")
    PATTERN_2 = re.compile(r"\^\s*|\s\s*")
    PATTERN_DIGITS = re.compile(r"\d")
    PATTERN_HTTPS = re.compile(r"https:?://\S+|www\.\S+")
    text = text.lower()  # lowercase text
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    text = re.sub(PATTERN_1, ' ', text)
    text = re.sub(PATTERN_2, ' ', text)
    text = re.sub(PATTERN_DIGITS, ' ', text)
    text = re.sub(PATTERN_HTTPS, ' ', text)
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    tokens = [w for w in tokens if (len(w) > min_words and len(w) < max_words)]
    # remove new stopwords from the token list
    tokens = [w for w in tokens if w not in stopwords]
    return tokens


def clean_sentences(df):
    """
    Remove irrelavant characters (in new column clean_sentence).
    Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
    """
    print('Cleaning sentences...')
    df['clean_sentence'] = df['sn_short_description'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    return df


df = clean_sentences(df)
pd.options.display.max_colwidth = 500

df.head(3)



