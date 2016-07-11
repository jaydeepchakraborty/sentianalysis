
##reading positive and negative mark_safe statements
pos_file_name = "pos_backup.txt"
neg_file_name = "neg_backup.txt"
test_file_name_fb = "fb_posts_backup.txt"
test_file_name_twitter = "twitter_posts_backup.txt"
stop_word_file_name = "stopword"
feture_file_name = "features_backup.txt"
train_mark_safe = []

for i in range(3):
    import nltk
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    with open(stop_word_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            stop_words.append(line)
    
    f.close()
    
    
    if not train_mark_safe:
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
    
    def get_word_features(wordlist):
        ##for now it is returning all the words, but in future we have to implement something 
        ##that will extract most valuable feature words
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features
    
    all_feture_words = []
    with open(feture_file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            if line not in stop_words:
                all_feture_words.append(line)
            
    f.close()
    
    word_features = get_word_features(all_feture_words)
    
    
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
    feature_results = []
    test_data = []
    num_ind = 0
    train_mark_safe = []
    for post in predict_mark_safe:
        senti_result = classifier.classify(extract_features(post.split()))
        filtered_words = [w for w in post.split() if not w in stop_words]
        feature_results.append(filtered_words)
        if(senti_result == "positive"):
            pos_results.append((post,senti_result))
            if(num_ind <= 10):
                test_data.append((post,senti_result))
            else:
                train_mark_safe.append((post,senti_result))
        elif(senti_result == "negative"):
            neg_results.append((post,senti_result))
            if(num_ind <= 10):
                test_data.append((post,senti_result))
            else:
                train_mark_safe.append((post,senti_result))
    
        num_ind = num_ind +1
        
        
    with open(pos_file_name, "a", encoding='utf-8') as f:
        for pos_val,sentiment in pos_results:
            f.write(pos_val)
            f.write("\n")
        
    f.close()
    
    with open(neg_file_name, "a", encoding='utf-8') as f:
        for neg_val,sentiment in neg_results:
            f.write(neg_val)
            f.write("\n")
        
    f.close()
    
    with open(feture_file_name, "a", encoding='utf-8') as f:
        for feature_val in feature_results:
            f.write(feature_val)
            f.write("\n")
        
    f.close()
    
    file = open("result.txt", "w", encoding='utf-8')
    
    for result in pos_results:
        file.write(str(result)+"\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    for result in neg_results:
        file.write(str(result)+"\n")
        file.write("--------------------------------------------------------------------------------------------\n")
    
    file.close()
    
    
        