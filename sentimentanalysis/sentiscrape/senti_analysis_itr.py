from collections import defaultdict
import nltk
import time
import logging
import sys
from nltk.corpus import stopwords
from nltk.metrics.scores import precision, recall, f_measure


# #reading positive and negative mark_safe statements
topic_name = "topics/marksafe/"
log_file_name = 'topics/marksafe/log/sentiment_analysis_log.log'
pos_file_name = topic_name+"pos_init_train.txt"
neg_file_name = topic_name+"neg_init_trin.txt"
test_file_name_fb = topic_name+"fb_posts_init_train.txt"
test_file_name_twitter = topic_name+"twitter_posts_init_train.txt"
stop_word_file_name = topic_name+"stopword"
test_file_name = topic_name+"test"
ref_file_name = topic_name+"reference"
pos_train_file_name = topic_name+"pos_train"
neg_train_file_name = topic_name+"neg_train"
result_init = topic_name+"result_init.txt"

def populateStopWords(root_logger):
    start_time = time.time()
    stop_words = stopwords.words('english')
    with open(stop_word_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').lower()
            stop_words.append(line)
    
    f.close()
    root_logger.debug("populateStopWords method :- "+str((time.time() - start_time)) + " seconds")
    return stop_words

def populateInitTrainSet(root_logger):
    start_time = time.time()
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
    root_logger.debug("init train data length:- "+str(len(train_mark_safe)))
    root_logger.debug("populateInitTrainSet method :- "+str((time.time() - start_time)) + " seconds")
    return train_mark_safe
    

def populateTrainSet(sample_val,root_logger):
    start_time = time.time()
    train_mark_safe = []
    with open(pos_train_file_name+sample_val+".txt", "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            train_mark_safe.append((line.split(), "positive"))
    
    f.close()
        
    with open(neg_train_file_name+sample_val+".txt", "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            train_mark_safe.append((line.split(), "negative"))
            
    f.close()
#     shuffle(train_mark_safe)
    root_logger.debug("train data length:- "+str(len(train_mark_safe)))
    root_logger.debug("populateTrainSet method :- "+str((time.time() - start_time)) + " seconds")
    return train_mark_safe


def populateInitTestingSet(root_logger):
    start_time = time.time()
    test_mark_safe = []
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
    root_logger.debug("init test data length:- "+str(len(test_mark_safe))) 
    root_logger.debug("populateInitTestingSet method :- "+str((time.time() - start_time)) + " seconds") 
    return test_mark_safe

   
def populateTestingSet(itrVal,root_logger):
    start_time = time.time()
    test_mark_safe = []
    with open(test_file_name+itrVal+".txt", "r", encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                test_mark_safe.append(line)
                
    f.close()
    root_logger.debug("test data length:- "+str(len(test_mark_safe)))
    root_logger.debug("populateTestingSet method :- "+str((time.time() - start_time)) + " seconds") 
    return test_mark_safe

def populateFeatureWords(train_mark_safe,stop_words,root_logger):
    start_time = time.time()
    all_feature_words = []
    for words, sentiment in train_mark_safe:
        filtered_words = [w for w in words if not w.lower() in stop_words]
        all_feature_words.extend(filtered_words)
    root_logger.debug("feature data length:- "+str(len(all_feature_words)))
    root_logger.debug("populateFeatureWords method :- "+str((time.time() - start_time)) + " seconds")
    return all_feature_words

def get_word_features(wordlist,root_logger):
    # #for now it is returning all the words, but in future we have to implement something 
    # #that will extract most valuable feature words
    start_time = time.time()
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    root_logger.debug("feature data length:- "+str(len(word_features)))
    root_logger.debug("get_word_features method :- "+str((time.time() - start_time)) + " seconds")
    return word_features


def testInitData(classifier, test_mark_safe, stop_words,word_features,root_logger):
    start_time = time.time()
    pos_results = []
    neg_results = []
    
    def extract_features(document):
        document_words = set(document)
        features = {}
#         print(len(word_features))
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
            
        return features
    
    for test_post in test_mark_safe:
            refined_test_post_lst = [w for w in test_post.split() if not w.lower() in stop_words]
            senti_result = classifier.classify(extract_features(refined_test_post_lst))
            if(senti_result == "positive"):
                pos_results.append((test_post, senti_result))
            elif(senti_result == "negative"):
                neg_results.append((test_post, senti_result))
            
    root_logger.debug("testInitData method :- "+str((time.time() - start_time)) + " seconds")
    root_logger.debug("init positive data  :- "+str(len(pos_results)))
    root_logger.debug("init negative data  :- "+str(len(neg_results)))
    return pos_results, neg_results

def testData(classifier, test_mark_safe, stop_words,word_features,root_logger):
    start_time = time.time()
    pos_results = []
    neg_results = []
    
    def extract_features(document):
        document_words = set(document)
        features = {}
#         print(len(word_features))
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
            
        return features
    
    outsets = defaultdict(set)
    out_mark_safe = []
    i = 0
    for test_post in test_mark_safe:
            refined_test_post_lst = [w for w in test_post.split() if not w.lower() in stop_words]
            senti_result = classifier.classify(extract_features(refined_test_post_lst))
            if(senti_result == "positive"):
                pos_results.append((test_post, senti_result))
            elif(senti_result == "negative"):
                neg_results.append((test_post, senti_result))
            
            out_mark_safe.append((test_post.split(), senti_result))
            outsets[senti_result].add(i)
            i = i+1
    root_logger.debug("testData method :- "+str((time.time() - start_time)) + " seconds") 
    root_logger.debug("positive data  :- "+str(len(pos_results)))
    root_logger.debug("negative data  :- "+str(len(neg_results)))       
    return pos_results, neg_results, outsets, out_mark_safe

def updateData(pos_results , neg_results,root_logger):
    start_time = time.time()
    opt_test_pos_val_ind = int((len(pos_results) * 0.2))
    opt_test_neg_val_ind = int((len(neg_results) * 0.2))
    
    ## as we are doing 20:80, so the number of sample will be 5
    for sample_val in range(1,6):
        start_test_pos_val = opt_test_pos_val_ind * (sample_val - 1)
        end_test_pos_val = opt_test_pos_val_ind * (sample_val)
        start_test_neg_val = opt_test_neg_val_ind * (sample_val - 1)
        end_test_neg_val = opt_test_neg_val_ind * (sample_val)
        
        test_file = open(test_file_name+str(sample_val)+".txt", "w", encoding='utf-8')
        ref_file = open(ref_file_name+str(sample_val)+".txt", "w", encoding='utf-8')
    
        #myList[2:5] third element to the fifth, from myList[2] to myList[4]
        extracted_new_test_pos_data = pos_results[start_test_pos_val:end_test_pos_val]
        extracted_new_test_neg_data = neg_results[start_test_neg_val:end_test_neg_val]
    
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
        ref_file.close()
        
        pos_train_file = open(pos_train_file_name+str(sample_val)+".txt", "w", encoding='utf-8')
        ind = 0
        for pos_val, sentiment in pos_results:
            if not (ind>=start_test_pos_val and ind<end_test_pos_val):
                pos_train_file.write(str(pos_val))
                pos_train_file.write("\n")
            ind = ind +1
        pos_train_file.close()
      
        ind = 0
        neg_train_file = open(neg_train_file_name+str(sample_val)+".txt", "w", encoding='utf-8')
        for neg_val, sentiment in neg_results:
            if not (ind>=start_test_neg_val and ind<end_test_neg_val):
                neg_train_file.write(str(neg_val))
                neg_train_file.write("\n") 
            ind = ind +1
        neg_train_file.close()
    root_logger.debug("updateData method :- "+str((time.time() - start_time)) + " seconds") 

def writeResult(pos_results , neg_results, root_logger):
    start_time = time.time()
    file = open(result_init, "w", encoding='utf-8')
    
    for result in pos_results:
        file.write(str(result))
        file.write("\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    for result in neg_results:
        file.write(str(result))
        file.write("\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    root_logger.debug("writeResult method :- "+str((time.time() - start_time)) + " seconds")
    file.close()

def mainMtdh():
    
    root_logger= logging.getLogger("sentiment_analysis")
    root_logger.setLevel(logging.DEBUG) 
    handler = logging.FileHandler(log_file_name, 'w', 'latin1') # or whatever
    handler.setFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    root_logger.addHandler(handler)
    root_logger.debug('start calling mainMtdh method')
    main_mtdh_start_time = time.time()
    try:
        stop_words = populateStopWords(root_logger)
            
        ##code for init  start--------------------------------------------------------------------------    
        train_mark_safe = populateInitTrainSet(root_logger)  # populating train_mark_safe[]
        test_mark_safe = populateInitTestingSet(root_logger)  # populating test_mark_safe[]
        all_feature_words = populateFeatureWords(train_mark_safe,stop_words,root_logger)  # populating all_feature_words[]
        word_features = get_word_features(all_feature_words,root_logger)  # # populating word_features[]
             
        def extract_features(document):
            document_words = set(document)
            features = {}
    #         print(len(word_features))
            for word in word_features:
                features['contains(%s)' % word] = (word in document_words)
                
            return features
             
        # #from train_mark_safe create training_set to train the model
        start_time = time.time()
        training_set = nltk.classify.apply_features(extract_features, train_mark_safe)
        root_logger.debug("init nltk.classify.apply_features :- "+str((time.time() - start_time)) + " seconds")
        start_time = time.time()
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        root_logger.debug("init nltk.NaiveBayesClassifier.train :- "+str((time.time() - start_time)) + " seconds")
             
        pos_results , neg_results = testInitData(classifier, test_mark_safe, stop_words, word_features,root_logger)
                 
        updateData(pos_results , neg_results,root_logger)
        
        writeResult(pos_results , neg_results,root_logger) #just to store the result
        ##code for init  end--------------------------------------------------------------------------    
        
        
        
        #code for iteration 2 for sample1,2,3,4,5 start-------------------------------------------------------------------------- 
        for sample_val in range(1,6):
            root_logger.debug("Sample val:- "+ str(sample_val))
            train_mark_safe = populateTrainSet(str(sample_val),root_logger)# populating train_mark_safe[]
            test_mark_safe = populateTestingSet(str(sample_val),root_logger)# populating test_mark_safe[]
            all_feature_words = populateFeatureWords(train_mark_safe,stop_words,root_logger)# populating all_feature_words[]
            word_features = get_word_features(all_feature_words,root_logger)## populating word_features[]
                
            def extract_features(document):
                    document_words = set(document)
                    features = {}
            #         print(len(word_features))
                    for word in word_features:
                        features['contains(%s)' % word] = (word in document_words)
                    return features
            ##from train_mark_safe create training_set to train the model
            start_time = time.time()
            training_set = nltk.classify.apply_features(extract_features, train_mark_safe)
            root_logger.debug("sample"+str(sample_val)+" train nltk.classify.apply_features :- "+str((time.time() - start_time)) + " seconds")
            start_time = time.time()
            classifier = nltk.NaiveBayesClassifier.train(training_set)
            root_logger.debug("sample"+str(sample_val)+" nltk.NaiveBayesClassifier.train :- "+str((time.time() - start_time)) + " seconds")
            pos_results ,neg_results, outsets , out_mark_safe = testData(classifier,test_mark_safe,stop_words,word_features,root_logger)
            refsets = defaultdict(set)
            with open(ref_file_name+str(sample_val)+".txt", "r", encoding='utf-8') as f:
                i = 0
                for line in f:
                    refsets[line.split()[-1]].add(i)
                    i = i +1
            f.close() 
            
            start_time = time.time()
            out_set = nltk.classify.apply_features(extract_features, out_mark_safe)
            root_logger.debug("sample"+str(sample_val)+" output nltk.classify.apply_features :- "+str((time.time() - start_time)) + " seconds")
            
            start_time = time.time()
            pos_precision = precision(refsets['positive'], outsets['positive'])
            pos_recall = recall(refsets['positive'], outsets['positive'])
            pos_fmeasure = f_measure(refsets['positive'], outsets['positive'])
            neg_precision = precision(refsets['negative'], outsets['negative'])
            neg_recall = recall(refsets['negative'], outsets['negative'])
            neg_fmeasure = f_measure(refsets['negative'], outsets['negative'])
            accuracy = nltk.classify.accuracy(classifier,out_set)*100
            
            root_logger.debug('pos precision:' +str(pos_precision))
            root_logger.debug('pos recall:'+ str(pos_recall))
            root_logger.debug('pos F-measure:'+ str(pos_fmeasure))
            root_logger.debug('neg precision:'+ str(neg_precision))
            root_logger.debug('neg recall:'+ str(neg_recall))
            root_logger.debug('neg F-measure:'+str(neg_fmeasure))
            
            root_logger.debug('accuracy '+ str(accuracy)) 
            root_logger.debug("time to calculate precision and recal for sample "+str(sample_val)+str((time.time() - start_time)) + " seconds") 
        #code for iteration 2 for sample1,2,3,4,5 end--------------------------------------------------------------------------    
    except Exception as e:
            root_logger.debug("error:- -------------------------------------")
            root_logger.error(e, exc_info=True)   
    root_logger.debug("total time :- "+str((time.time() - main_mtdh_start_time)) + " seconds")
    root_logger.debug('end calling mainMtdh method')

mainMtdh()