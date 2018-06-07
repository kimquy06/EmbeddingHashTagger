import pickle
import numpy as np
import scipy.spatial.distance as distance
import scipy
from gensim.models import Word2Vec


dimension_output = 100


def create_ht_embedding_table(ht_to_embedding):
    '''return table with embeddings and index of ht in table'''
    ht_embedding_table = np.ndarray(shape=(len(ht_to_embedding.keys()), dimension_output), dtype=np.float32)
    i = 0
    idx_to_ht = {}
    for ht, em in ht_to_embedding.items():
        ht_embedding_table[i] = em
        idx_to_ht[i] = ht
        i += 1
    return ht_embedding_table, idx_to_ht


def f1_aloc(test_set, test_predictions):
    i = 0
    precision = 0
    recall = 0
    for prediction in test_predictions:
        counter , pred_ht  = prediction
        tweet = test_set[counter]
        for ht in tweet["hashtags"]:
            if pred_ht.lower() == ht.lower():
                precision += 1
                recall += 1
                break
    print(precision)

    precision /= len(test_set)
    recall /= len(test_set)
    return (2 * precision * recall) / (precision + recall)


def create_tweet_avg_embedding(dct): # this is for avg embedding for tweet
    tweet_average_ambedding = {}
    for key in dct: # d is a dict
        embedding_sum = np.zeros(shape=dimension_output)
        d = dct[key]
        for word in d:
            embedding_sum += d[word]
        tweet_average_ambedding[key] = embedding_sum
    return tweet_average_ambedding
###############################################################################
###############################################################################
###############################################################################

file_object = open("all_preproc_no_hashtags_really.pickle", 'rb')

file_desc = pickle.load(file_object)
tweets_contain_hashtag = dict()  # dict: hashtag -> tweetS contain the hashtag

for tweet_info in file_desc[0 : int(len(file_desc) * .9995)]:
#for tweet_info in file_desc[:5000]:
    for hashtag in tweet_info['hashtags']:
        hashtag_lower = hashtag.lower()
        document = tweet_info['text']
        if hashtag_lower in tweets_contain_hashtag:
            temp_str = tweets_contain_hashtag[hashtag_lower]
            temp_str += document
            tweets_contain_hashtag[hashtag_lower] = temp_str
        else:
            tweets_contain_hashtag[hashtag_lower] = document

embedding_of_hashtag = dict() # dict(hashtag -> dict(word -> embedding)), this is for testing part
embedding_of_words = dict()
hashtag_num = -1

for hashtag in tweets_contain_hashtag:
    hashtag_num += 1
    embedding_of_words[hashtag] = dict()
    temp_dict = dict()
    sentence_list = []
    sentence = tweets_contain_hashtag[hashtag].split(' ')  # = Dh
    sentence_list.append(sentence)  # just for word2vec argument
    vector_w_h = Word2Vec(sentence_list, min_count=1,
                          window=3, sg=1)  # dimension default = 100
    embedding_sum_avg = np.zeros(shape=dimension_output)
    for word in sentence:
        embedding_sum_avg += vector_w_h[word]
        temp_dict[word] = vector_w_h[word]
    embedding_of_words[hashtag] = temp_dict
    embedding_of_hashtag[hashtag] = (embedding_sum_avg / len(sentence))




    #words = list(vector_w_h.wv.vocab)
    # print(words)
    # print(type(vector_w_h[sentence[0]]))

ht_embedding_table = create_ht_embedding_table(embedding_of_hashtag)[0]
#tweet_average_ambedding = create_tweet_avg_embedding(embedding_of_words)

print("########################################################################")
print("###########################  TRAINING READY  ###########################")
print("########################################################################")

# TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING

max_aligned_hashtag = ""
max_alignment_score = 0
top_hashtags_num = 1
top_hashtags = []
hashtags_in_test_tweets = dict()
test_predictions = [] # for f1 measure
tweet_embedding = np.empty(shape=(1,dimension_output))


test_set995 = file_desc[int(len(file_desc) * .9995):]
test_predictions995 = []
tweet_counter = -1
#for tweet_info in file_desc[int(len(file_desc) * .80):]:
#for tweet_info in file_desc[int(len(file_desc) * .995):]:
for tweet_info in test_set995:
    tt = tweet_info['text']
    wt = tweet_info['text'].split(' ')
    tweet_counter += 1

    for hashtag in tweets_contain_hashtag:
        wh = tweets_contain_hashtag[hashtag].split(' ') # set of words across all tweets that contain h
        emb_sum_avg = np.zeros(shape=dimension_output)

        c=0
        vector_wt_h = embedding_of_words[hashtag]
        for word in wt:
            if word in vector_wt_h:
                emb_sum_avg += vector_wt_h[word]
                c += 1
        
        avg_embedding_of_tweet = (emb_sum_avg / c)

        #avg_embedding_of_tweet = tweet_average_ambedding[hashtag]
        tweet_embedding = np.array([avg_embedding_of_tweet])
        h_similarity = scipy.spatial.distance.cdist(ht_embedding_table,tweet_embedding.reshape(1, -1), 'cosine')

        if max(h_similarity) > max_alignment_score:
            max_aligned_hashtag = hashtag
            max_alignment_score = max(h_similarity)

    g = max_aligned_hashtag

    list_of_hashtags_for_topk = []

    if top_hashtags_num == 1:
        top_hashtags.append(g)
        print(tt ," with HASHTAG: ", g)
        test_predictions995.append((tweet_counter, g))
    else:  # top K hashtags
        for tweet in file_desc[int(len(file_desc) * .80):]:
            if g in tweet['hashtags']:
                list_of_hashtags_for_topk += tweet['hashtags']
        avg_emb = np.zeros(shape=dimension_output)
        emb_words = embedding_of_words[g]
        count = 0
        for w in wt:
            if w in emb_words:
                avg_emb += emb_words[w]
                count += 1
        avg_emb_final = avg_emb / count

        list_of_emb = []
        for i in list_of_hashtags_for_topk:
            if i in embedding_of_hashtag:
                # print(embedding_of_hashtag)
                cosine_sim = 1 -  distance.cosine(avg_emb_final, embedding_of_hashtag[i])
                list_of_emb.append(cosine_sim)

        zip_list_ht_emb = list(zip(list_of_hashtags_for_topk, list_of_emb))
        zip_list_ht_emb.sort(key=lambda t: t[1])
        
        print("TOP K HT: ", zip_list_ht_emb[:top_hashtags_num])
        print(tt ," with HASHTAGs: ", max_aligned_hashtag)

print("F1: ", f1_aloc(test_set995, test_predictions995))