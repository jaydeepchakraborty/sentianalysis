from gensim import corpora, models

class MainCls:
        
    
    def callCreateProbEq(self):
        import logging
        import sys
        import threading
        from threading import Thread
        
        
        root_logger= logging.getLogger("recommendation")
        root_logger.setLevel(logging.DEBUG) 
        handler = logging.FileHandler('recoinfo.log', 'w', 'latin1') # or whatever
        handler.setFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
        root_logger.addHandler(handler)
        root_logger.debug('start calling callCreateProbEq method of mainCls class')

        createProbEq = CreateProbEq()
        
        try:
            Thread(target = createProbEq.writeProbEq('twitter_posts.txt',"twitter_eq.txt")).start()
        except:
            root_logger.debug("error in creating probability equation for professor -:")
            root_logger.error(sys.exc_info()[0])
            pass
            

        root_logger.debug('end calling callCreateProbEq method of mainCls class')
        
class CreateProbEq:
    
    def writeProbEq(self,abstract_file_nm,prob_eq_file_nm):
        print(abstract_file_nm,prob_eq_file_nm)
        file = open(abstract_file_nm, 'rt', encoding='latin1')
        
        documents = [file.read()]
        
        import nltk
        stoplist = []
        stoplist = nltk.corpus.stopwords.words('english')
        f = open('stopword','rt', encoding='latin1')
        for word in f.read().split():
            stoplist.append(word)
            
        from nltk.stem import PorterStemmer, WordNetLemmatizer
   
        #port = PorterStemmer()
        #stemmed_doc = " ".join([port.stem(i) for i in documents[0].lower().split()])
        
        wnl = WordNetLemmatizer()
        lemmatized_doc = " ".join([wnl.lemmatize(i) for i in documents[0].lower().split()])
        #print(lemmatized_doc)
        
        texts = [[word for word in lemmatized_doc.split() if word not in stoplist]]
        
        #print(texts)
        all_tokens = sum(texts, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        
        texts = [[word for word in text if word not in tokens_once] for text in texts]

        # print(texts)
        dictionary = corpora.Dictionary(texts)
        
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        text_file = open(prob_eq_file_nm, "w", encoding='latin1')
        
#-----------------------------------------------------------------------------------------------------------------   
        lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, 
                                       update_every=1, chunksize=10000, passes=1)
        
        for model in lda_model.print_topics():
            text_file.write(str(model)+"\n")
    
#-----------------------------------------------------------------------------------------------------------------   
        hdp_model = models.hdpmodel.HdpModel(corpus, id2word=dictionary)
        # print(hdp_model.show_topics(3))
        for model in hdp_model.print_topics():
            text_file.write(str(model)+"\n")
        
#-----------------------------------------------------------------------------------------------------------------   
        lsa_model = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=3)
        for model in lsa_model.print_topics():
            text_file.write(str(model)+"\n")
            
        
        text_file.close()
        file.close()

        
mainCls = MainCls()
mainCls.callCreateProbEq()