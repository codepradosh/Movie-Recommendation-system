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
