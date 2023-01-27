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

