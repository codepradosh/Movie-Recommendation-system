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
