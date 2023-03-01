#Word 2 Vec

start_time = time.time()

from gensim.models.word2vec import Word2Vec

# Create model
word2vec_model = Word2Vec(min_count=0, workers = 8, vector_size=300) 
# Prepare vocab
word2vec_model.build_vocab(df.tok_lem_sentence.values)
# Train
word2vec_model.train(df.tok_lem_sentence.values, total_examples=word2vec_model.corpus_count, epochs=100)

end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))





#Prediction

def is_word_in_model(word, model):
    """
    Check on individual words ``word`` that it exists in ``model``.
    """
    assert type(model).__name__ == 'KeyedVectors'
    is_in_vocab = word in model.key_to_index.keys()
    return is_in_vocab

def predict_w2v(query_sentence, dataset, model, k=10):
    query_sentence = query_sentence.split()
    in_vocab_list, best_index = [], [0]*k
    for w in query_sentence:
        # remove unseen words from query sentence
        if is_word_in_model(w, model.wv):
            in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance
    if len(in_vocab_list) > 0:
        sim_mat = np.zeros(len(dataset))  # TO DO
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                sim_sentence = model.wv.n_similarity(
                        in_vocab_list, data_sentence)
            else:
                sim_sentence = 0
            sim_mat[i] = np.array(sim_sentence)
        # Take the five highest norm
        best_index = np.argsort(sim_mat)[::-1][:k]
    return best_index

# Predict
start_time = time.time()
best_index = predict_w2v(query_sentence, df['tok_lem_sentence'].values, word2vec_model)

display(df[['sn_short_description']].iloc[best_index])
end_time = time.time()

print("Time taken: {:.2f} seconds".format(end_time - start_time))




start_time = time.time()
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))



import re
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

def clean_text(text):
    # Define regular expressions for data pre-processing
    PATTERN_S = re.compile("\'s")
    PATTERN_RN = re.compile("\\r\\n")
    PATTERN_PUNC = re.compile(r"[^\w\s]")
    PATTERN_1 = re.compile(r"\b(?<!-)[0-9]+\b\s*")
    PATTERN_2 = re.compile(r"^\s*|\s\s*")
    PATTERN_DIGITS = re.compile(r"\d")
    PATTERN_HTTPS = re.compile(r"https?://\S+|www\.\S+")
    PATTERN_HTML = re.compile(r'<.*?>')
    
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(PATTERN_HTTPS, ' ', text)
    text = re.sub(PATTERN_HTML, ' ', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    text = re.sub(PATTERN_1, ' ', text)
    text = re.sub(PATTERN_DIGITS, ' ', text)
    text = re.sub(PATTERN_2, ' ', text)
    text = re.sub(r'icto\s+(\b[0-9]+\b)\s*', r'icto-\1', text).strip()
    
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w, pos='v') for w in word_tokenize(sentence)]
        tokens = [stemmer.lemmatize(w, pos='a') for w in tokens]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    tokens = [w for w in tokens if (len(w) > min_words and len(w) < max_words and w not in stopwords)]
    return tokens

def clean_sentences(df):
    print('Cleaning Sentences...')
    df['clean_sentence'] = df['sn_short_description'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    
    return df

def extend_nltk_stopwordlist():
    new_stopwords = ['esc2l1', 'crit', 'warn', 'val', 'percent', 'pct', 'severity', 'warning', 'rule', 'initial', 'alert']
    remove_stopwords = ['am', 'not', 'down']
    stopwrd = stopwords.words('english')
    stopwrd.extend(new_stopwords)
    final_stop_words = set([word for word in stopwrd if word not in remove_stopwords])
    
    return final_stop_words

def remove_english_stopwords(text):
    stopwrds = extend_nltk_stopwordlist()
    t = [token for token in text.split() if token not in stop


         
         
         
from sklearn.metrics.pairwise import cosine_similarity

def predict_w2v(query_sentence, dataset, model, k=10):
    query_sentence = query_sentence.split()
    in_vocab_list, best_index = [], [0]*k
    for w in query_sentence:
        # remove unseen words from query sentence
        if is_word_in_model(w, model.wv):
            in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance
    if len(in_vocab_list) > 0:
        sim_mat = np.zeros(len(dataset))
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                sim_sentence = cosine_similarity(
                    [model.wv[w] for w in in_vocab_list], 
                    [model.wv[w] for w in data_sentence]
                ).mean()
            else:
                sim_sentence = 0
            sim_mat[i] = np.array(sim_sentence)
        # Take the k highest similarity scores
        best_index = np.argsort(sim_mat)[::-1][:k]
    return best_index
         
        

final_outcome = 'Please refer below confluence document to address this issue:<br><br>'
final_outcome += '<a href="' + outcome['URL'] + '">' + outcome['Recommended Document'] + '</a>'
