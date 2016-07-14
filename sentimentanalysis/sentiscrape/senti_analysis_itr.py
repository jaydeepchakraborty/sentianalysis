import nltk
from nltk.corpus import stopwords
from random import shuffle
from matplotlib.delaunay.testfuncs import TestData
from collections import defaultdict
from nltk.metrics.scores import precision,recall,f_measure


# #reading positive and negative mark_safe statements
pos_file_name = "pos_backup.txt"
neg_file_name = "neg_backup.txt"
test_file_name_fb = "fb_posts_backup.txt"
test_file_name_twitter = "twitter_posts_backup.txt"
stop_word_file_name = "stopword"
test_file_name = "test"
ref_file_name = "reference"
pos_train_file_name = "pos_train"
neg_train_file_name = "neg_train"

def populateStopWords():
    stop_words = stopwords.words('english')
    with open(stop_word_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            stop_words.append(line)
    
    f.close()
    return stop_words

def populateTrainSet():
    train_mark_safe = []
    with open(pos_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            train_mark_safe.append((line.split(), "positive"))
    
    f.close()
        
    with open(neg_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            train_mark_safe.append((line.split(), "negative"))
            
    f.close()
#     shuffle(train_mark_safe)
    return train_mark_safe
    
    
def populateTestingSet(itrVal):
    test_mark_safe = []
    if(itrVal == 0):
        print("iteration 0")
        post_str = ""
        with open(test_file_name_fb, "r", encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                if("---------" not in line):
                    post_str = post_str + " " + line
                else:
                    test_mark_safe.append(post_str)
                    post_str = ""
                
        f.close()
    
        post_str = ""
        with open(test_file_name_twitter, "r", encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                if("---------" not in line):
                    post_str = post_str + " " + line
                else:
                    test_mark_safe.append(post_str)
                    post_str = ""
                
        f.close()
    elif(itrVal != 0):
        with open(test_file_name+itrVal+".txt", "r", encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                test_mark_safe.append(line)
                
        f.close()

    
    return test_mark_safe

def populateFeatureWords(train_mark_safe):
    all_feature_words = []
    for words, sentiment in train_mark_safe:
        filtered_words = [w for w in words if not w.lower() in stop_words]
        all_feature_words.extend(filtered_words)
    return all_feature_words

def get_word_features(wordlist):
    # #for now it is returning all the words, but in future we have to implement something 
    # #that will extract most valuable feature words
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features




def testInitData(classifier, test_mark_safe):
    pos_results = []
    neg_results = []
    
    for test_post in test_mark_safe:
            senti_result = classifier.classify(extract_features(test_post.split()))
            refined_test_post_lst = [w for w in test_post.split() if not w.lower() in stop_words]
            refined_test_post = ""
            for refined_test_post_val in refined_test_post_lst:
                refined_test_post = refined_test_post + " " + refined_test_post_val
            if(senti_result == "positive"):
                pos_results.append((refined_test_post, senti_result))
            elif(senti_result == "negative"):
                neg_results.append((refined_test_post, senti_result))
            
    return pos_results, neg_results

def testData(classifier, test_mark_safe):
    pos_results = []
    neg_results = []
    
    from collections import defaultdict
    outsets = defaultdict(set)
    i = 0
    for test_post in test_mark_safe:
            senti_result = classifier.classify(extract_features(test_post.split()))
            refined_test_post_lst = [w for w in test_post.split() if not w.lower() in stop_words]
            refined_test_post = ""
            for refined_test_post_val in refined_test_post_lst:
                refined_test_post = refined_test_post + " " + refined_test_post_val
            if(senti_result == "positive"):
                pos_results.append((refined_test_post, senti_result))
            elif(senti_result == "negative"):
                neg_results.append((refined_test_post, senti_result))
            
            outsets[senti_result].add(i)
            i = i+1
            
    return pos_results, neg_results, outsets

def updateData(pos_results , neg_results,sampleVal):
    
    test_file = open(test_file_name+sampleVal+".txt", "w", encoding='utf-8')
    ref_file = open(ref_file_name+sampleVal+".txt", "w", encoding='utf-8')
    
    opt_test_pos_val_ind = int((len(pos_results) * 0.2))
    opt_test_neg_val_ind = int((len(neg_results) * 0.2))
    
    ##first 20% data
    extracted_new_test_pos_data = pos_results[:opt_test_pos_val_ind]
    extracted_new_test_neg_data = neg_results[:opt_test_neg_val_ind]
    
    for pos_val, sentiment in extracted_new_test_pos_data:
        ref_file.write(str(pos_val)+", "+str(sentiment))
        ref_file.write("\n")
        test_file.write(str(pos_val))
        test_file.write("\n")
    for neg_val, sentiment in extracted_new_test_neg_data:
        ref_file.write(str(neg_val)+", "+str(sentiment))
        ref_file.write("\n")
        test_file.write(str(neg_val))
        test_file.write("\n") 
    test_file.close()  
    
    ##next 80% data
    extracted_new_train_pos_data = pos_results[opt_test_pos_val_ind:]
    extracted_new_train_neg_data = neg_results[opt_test_neg_val_ind:]
    
    pos_train_file = open(pos_train_file_name+sampleVal+".txt", "w", encoding='utf-8')
    neg_train_file = open(neg_train_file_name+sampleVal+".txt", "w", encoding='utf-8')
    
    for pos_val, sentiment in extracted_new_train_pos_data:
        pos_train_file.write(str(pos_val))
        pos_train_file.write("\n")
    pos_train_file.close()
    for neg_val, sentiment in extracted_new_train_neg_data:
        neg_train_file.write(str(neg_val))
        neg_train_file.write("\n") 
    neg_train_file.close()


     

def updateResult(pos_results , neg_results, itrVal):
    file = open("result" + str(itrVal) + ".txt", "w", encoding='utf-8')
    
    for result in pos_results:
        file.write(str(result))
        file.write("\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    for result in neg_results:
        file.write(str(result))
        file.write("\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    
    file.close()
        
##code for iteration 1 start--------------------------------------------------------------------------    
# stop_words = populateStopWords()
# print("stop words: - " + str(len(stop_words)))
# train_mark_safe = populateTrainSet()  # populating train_mark_safe[]
# print("train data: - " + str(len(train_mark_safe)))
# test_mark_safe = populateTestingSet(0)  # populating test_mark_safe[]
# print("test data: - " + str(len(test_mark_safe)))
# all_feature_words = populateFeatureWords(train_mark_safe)  # populating all_feature_words[]
# print("feature words: - " + str(len(all_feature_words)))
# word_features = get_word_features(all_feature_words)  # # populating word_features[]
#     
# def extract_features(document):
#         document_words = set(document)
#         features = {}
# #         print(len(word_features))
#         for word in word_features:
#             features['contains(%s)' % word] = (word in document_words)
#            
#         return features
#     
# # #from train_mark_safe create training_set to train the model
# training_set = nltk.classify.apply_features(extract_features, train_mark_safe)
# classifier = nltk.NaiveBayesClassifier.train(training_set)
#     
# pos_results , neg_results = testInitData(classifier, test_mark_safe)
#     
# print("positive:- " + str(len(pos_results)))
# print("negative:- " + str(len(neg_results)))
#         
# updateData(pos_results , neg_results, "1")
  
##code for iteration 1 end--------------------------------------------------------------------------    



#code for iteration 2 start--------------------------------------------------------------------------    
stop_words = populateStopWords()
print("stop words: - "+str(len(stop_words)))
train_mark_safe = populateTrainSet()# populating train_mark_safe[]
print("train data: - "+str(len(train_mark_safe)))
test_mark_safe = populateTestingSet("1")# populating test_mark_safe[]
print("test data: - "+str(len(test_mark_safe)))
all_feature_words = populateFeatureWords(train_mark_safe)# populating all_feature_words[]
print("feature words: - "+str(len(all_feature_words)))
word_features = get_word_features(all_feature_words)## populating word_features[]
    
def extract_features(document):
        document_words = set(document)
        features = {}
#         print(len(word_features))
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
           
        return features
##from train_mark_safe create training_set to train the model
training_set = nltk.classify.apply_features(extract_features, train_mark_safe)
classifier = nltk.NaiveBayesClassifier.train(training_set)
    
pos_results ,neg_results, outsets = testData(classifier,test_mark_safe)
print("positive:- "+str(len(pos_results)))
print("negative:- "+str(len(neg_results)))
 
refsets = defaultdict(set)
temp_ref_lst = []
with open(ref_file_name+"1"+".txt", "r", encoding='utf-8') as f:
    i = 0
    for line in f:
        refsets[line.split()[-1]].add(i)
        i = i +1
f.close()
     
# print(outsets) 
# print(refsets) 

print('pos precision:', precision(refsets['positive'], outsets['positive']))
print('pos recall:', recall(refsets['positive'], outsets['positive']))
print('pos F-measure:', f_measure(refsets['positive'], outsets['positive']))
print('neg precision:', precision(refsets['negative'], outsets['negative']))
print('neg recall:', recall(refsets['negative'], outsets['negative']))
print('neg F-measure:', f_measure(refsets['negative'], outsets['negative']))

print('accuracy ', nltk.classify.accuracy(classifier,training_set)*100)  
#code for iteration 2 end--------------------------------------------------------------------------    
