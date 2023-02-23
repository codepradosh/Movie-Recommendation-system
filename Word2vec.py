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




df = pd.read_csv(Boo)
df = df[~df.sn_short_description.isna()]
df = df.iloc[:20000]
df.head()
STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")  # matches `'s` from text`
PATTERN_RN = re.compile("\\r\\n") #matches `\r` and `\n`
PATTERN_PUNC = re.compile(r"[^\w\s]") # matches all non 0-9 A-z whitespace
PATTERN_1 = re.compile(r"\b(?<!-)[0-9]+\b\s*")
PATTERN_2 = re.compile(r"\^\s*|\s\s*")
PATTERN_DIGITS = re.compile(r"\d")
PATTERN_HTTPS = re.compile(r"https:?://\S+|www\.\S+")


def clean_text(text):
    """
    Series of cleaning. String to lower case, remove non words characters and numbers.
        text (str): input text
    return (str): modified initial text
    """
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
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                   and w not in stopwords)]
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


def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]
    return best_index

start_time = time.time()

token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
tfidf_mat = vectorizer.fit_transform(df['sn_short_description'].values) # -> (num_sentences, num_vocabulary)
end_time = time.time()

print("Time taken: {:.2f} seconds".format(end_time - start_time))
display(df)


