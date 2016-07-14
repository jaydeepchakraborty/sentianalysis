
##reading positive and negative mark_safe statements
pos_file_name = "pos.txt"
neg_file_name = "neg.txt"
test_file_name_fb = "fb_posts.txt"
test_file_name_twitter = "twitter_posts.txt"
stop_word_file_name = "stopword"

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
with open(stop_word_file_name, "r", encoding='utf-8') as f:
    for line in f:
        line = line.replace('\n','').lower()
        stop_words.append(line)

f.close()

print("stop words: - "+str(len(stop_words)))

train_mark_safe = []
with open(pos_file_name, "r", encoding='utf-8') as f:
    for line in f:
        line = line.replace('\n','')
        train_mark_safe.append((line.split(),"positive"))

f.close()

with open(neg_file_name, "r", encoding='utf-8') as f:
    for line in f:
        line = line.replace('\n','')
        train_mark_safe.append((line.split(),"negative"))
        
f.close()
print("train data: - "+str(len(train_mark_safe)))


predict_mark_safe = []

post_str = ""
with open(test_file_name_fb, "r", encoding='utf-8') as f:
    for line in f:
        line = line.replace('\n','')
        if("---------" not in line):
            post_str = post_str + line
        else:
            predict_mark_safe.append(post_str)
            post_str = ""
        
f.close()

post_str = ""
with open(test_file_name_twitter, "r", encoding='utf-8') as f:
    for line in f:
        line = line.replace('\n','')
        if("---------" not in line):
            post_str = post_str + line
        else:
            predict_mark_safe.append(post_str)
            post_str = ""
        
f.close()

print("test data: - "+str(len(predict_mark_safe)))

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        filtered_words = [w for w in words if not w.lower() in stop_words]
        all_words.extend(filtered_words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
#     print(word_features)
#     print(len(word_features)) 
    return word_features

all_words = get_words_in_tweets(train_mark_safe)
print("feature words: - "+str(len(all_words)))
word_features = get_word_features(all_words)


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features, train_mark_safe)
classifier = nltk.NaiveBayesClassifier.train(training_set)


pos_results = []
neg_results = []
for post in predict_mark_safe:
    senti_result = classifier.classify(extract_features(post.split()))
    if(senti_result == "positive"):
        pos_results.append((post,senti_result))
    elif(senti_result == "negative"):
        neg_results.append((post,senti_result))

print("positive:- "+str(len(pos_results)))
print("negative:- "+str(len(neg_results)))

file = open("result.txt", "w", encoding='utf-8')

for result in pos_results:
    file.write(str(result)+"\n")
    file.write("--------------------------------------------------------------------------------------------\n")

for result in neg_results:
    file.write(str(result)+"\n")
    file.write("--------------------------------------------------------------------------------------------\n")
file.close()


    