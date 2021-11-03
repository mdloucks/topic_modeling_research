# coding: utf-8
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from math import log10
import numpy as np
import networkx as nx
import pickle


with open("./preprocessed_bitcoin.pkl", 'rb') as f:
    data = pickle.load(f)

# convert the word_count list into a dictionary with words as keys, 
# and counts as values
voca2idx = {w: i for i, w in enumerate(data['voca'])}

voca = data['voca']

# HITS algorithm
hubs, user_score = nx.hits(data['user_network'], max_iter=500)
total_user_num = len(data['user_network'].nodes())
users = sorted(user_score, key=user_score.get, reverse=True)
score_sum = sum(user_score.values())

print("total users ", total_user_num)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Construct the term document matrix for the NMF topic modeling.
# 
# This first step constructs an m x n matrix where the number of rows is equal to the number
# of posts and the number of columns is equal to the number of unique words within the posts.
# because of this the matrix will be mostly sparse. 
#
# The eij(th) value in the matrix will be the number of times a word shows up in a given post
# ............................................................................................
tdm = dok_matrix((len(data['posts']), len(voca)), dtype=np.float32)
for i, post in enumerate(data['posts']):
    for word in post:
        tdm[i, voca2idx[word]] += 1
        
print("matrix shape", tdm.shape)
print("mat sample", tdm[0, 0], type(tdm[0, 0]))

# normalize the matrix along axis 1, which normalizes all of the word vectors
tdm = normalize(tdm)

# csr - compressed sparse row (efficient at performing row operations)
# initialized as dok for ease of initialization 
tdm = tdm.tocsr()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Some general commentary here on how NMF works for the sake of my own understanding
#
# This algorithm takes the term document matrix from above and approximately factors it into
# two different matrices w and h. w represents the topics and h represents the weights or
# significance of each of these topics. 
# Not entirely sure how to mathematics works, but because this factorization is an 
# approximation, it's necessary to have a cost function and try to minimize that as 
# much as possible using some machine learning techniques. 
# 
# The researchers used all of the default parameters for the NMF
# 
# This nmf variable is the one for the TOTAL community or all of the posts. Later we will 
# be using the same algorithm for majority users and opinion leaders
# ............................................................................................
K = 10
nmf = NMF(n_components=K, alpha=0.1, max_iter=500)
nmf.fit(tdm)
H_total = nmf.components_


# This portion is responsible for the graph seen on page 12
proportions = [0.0001, 0.0005, 0.001]
proportions.extend([i * 0.005 for i in range(1, 24)])
fwrite = open("./proportion_result.csv", "w")
fwrite.write("proportion, user_num, user_posts_num, similarity_avg\n")
for proportion in proportions:
    # 상위 유저 분석!
    top_num = round(len(users)*proportion)
    top_users = users[:top_num]

    user_posts = []
    for user in top_users:
        for post in data['user_posts'][user]:
            user_posts.append(post)
            
    top_user_posts_num = len(user_posts)

    tdm = dok_matrix((len(user_posts), len(voca)), dtype=np.float32)
    for i, post in enumerate(user_posts):
        for word in post:
            tdm[i, voca2idx[word]] += 1

    
    tdm = normalize(tdm)
    tdm = tdm.tocsr()

    # run a NMF for the the total original dataset
    nmf = NMF(n_components=K, alpha=0.1, max_iter=500)
    nmf.fit(tdm)
    H_top = nmf.components_


    # hungarian algorithm

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some commentary regarding the algorithm,
    #
    # The hungarian algorithm seeks to minimize the cost of a biparte graph where one set of 
    # vertices in the graph represents agents and the other represents tasks.
    # 
    # In this instance, the algorithm is trying to find the optimal match between the topics
    # generated from the majority and opinion leader grounds and the original set.
    # ............................................................................................

    top_distances = pairwise_distances(H_total, H_top, metric='cosine')
    _, top_indices = linear_sum_assignment(top_distances)

    similarity_average = 0
    for k in range(K):
        similarity = cosine_similarity(H_top[top_indices[k]].reshape(1, -1), H_total[k].reshape(1,-1))[0, 0] 
        similarity_average += similarity
        
    similarity_average /= K
    
    print("{}%, top_users: {}, top_user_posts: {}, top_similarity: {}".format(proportion*100, top_num, top_user_posts_num, similarity_average))    
    fwrite.write("{}, {}, {}, {}\n".format(proportion, top_num, top_user_posts_num, similarity_average))

fwrite.close()
