#sudo pip3 install nltk(natural language tool kit)
#----------------------------------------------------------------------------------
import nltk

#nltk.download("all")# download all the packages
#-----------------------------Tokenizer---------------------------------------------
from nltk.tokenize import sent_tokenize, word_tokenize

# EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
# 
# print(sent_tokenize(EXAMPLE_TEXT))
# print(word_tokenize(EXAMPLE_TEXT))
#-----------------------------StopWords---------------------------------------------

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# example_sent = "This is a sample sentence, showing off the stop words filtration."
# 
# stop_words = set(stopwords.words('english'))#get all words in english
# 
# word_tokens = word_tokenize(example_sent)
# 
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
# 
# print(filtered_sentence)

#-----------------------------Stemming words---------------------------------------------
from nltk.stem import PorterStemmer

# ps = PorterStemmer()
# example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
# 
# for w in example_words:
#     print(ps.stem(w))
#     
# new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
# words = word_tokenize(new_text)
# 
# for w in words:
#     print(ps.stem(w))
    
#-----------------------------wordnet---------------------------------------------

from nltk.corpus import wordnet

# syns = wordnet.synsets("program")
# 
# print(syns)
# print(syns[0].definition())
# print(syns[0].examples())
# 
# synonyms = []
# antonyms = []
# 
# for syn in wordnet.synsets("good"):
#     for lemma in syn.lemmas():
#         synonyms.append(lemma.name())
#         if lemma.antonyms():
#             antonyms.append(lemma.antonyms()[0].name())
# 
# print(set(synonyms))
# print(set(antonyms))

#-----------------------------similarity---------------------------------------------
# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("boat.n.01")
# print(w1.wup_similarity(w2))


#-----------------------------text classifier---------------------------------------------


#-------------------------------------------------------------------------------

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

#somehow remove unnecessary words and keep the needed ones.
train_tweets = [
    (['love', 'this', 'car'], 'positive'),
    (['this', 'view', 'amazing'], 'positive'),
    (['feel', 'great', 'this', 'morning'], 'positive'),
    (['excited', 'about', 'the', 'concert'], 'positive'),
    (['best', 'friend'], 'positive'),
    (['not', 'like', 'this', 'car'], 'negative'),
    (['this', 'view', 'horrible'], 'negative'),
    (['feel', 'tired', 'this', 'morning'], 'negative'),
    (['not', 'looking', 'forward', 'the', 'concert'], 'negative'),
    (['enemy'], 'negative')]

test_tweets = [
    (['feel', 'happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

#The list of word features need to be extracted from the tweets. 
#It is a list with every distinct words ordered by frequency of appearance.
all_words = get_words_in_tweets(train_tweets)
word_features = get_word_features(all_words)
#We end up with the following list of word features.
#Here ‘this’ is the most used word in our tweets, followed by ‘car’, followed by ‘concert".
#word_features = [
#     'this',
#     'car',
#     'concert',
#     'feel',
#     'morning',
#     'not',
#     'the',
#     'view',
#     'about',
#     'amazing',
#     ...
# ]

#To create a classifier, we need to decide what features are relevant. 
#To do that, we first need a feature extractor. The one we are going to use 
#returns a dictionary indicating what words are contained in the input passed. 
#Here, the input is the tweet. We use the word features list defined above along 
#with the input to create the dictionary.

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#As an example, let’s call the feature extractor with the document [‘love’, ‘this’, ‘car’]
# which is the first positive tweet. We obtain the following dictionary which indicates 
#that the document contains the words: ‘love’, ‘this’ and ‘car’.
# {'contains(not)': False,
#  'contains(view)': False,
#  'contains(best)': False,
#  'contains(excited)': False,
#  'contains(morning)': False,
#  'contains(about)': False,
#  'contains(horrible)': False,
#  'contains(like)': False,
#  'contains(this)': True,
#  'contains(friend)': False,
#  'contains(concert)': False,
#  'contains(feel)': False,
#  'contains(love)': True,
#  'contains(looking)': False,
#  'contains(tired)': False,
#  'contains(forward)': False,
#  'contains(car)': True,
#  'contains(the)': False,
#  'contains(amazing)': False,
#  'contains(enemy)': False,
#  'contains(great)': False}

training_set = nltk.classify.apply_features(extract_features, train_tweets)

#Now that we have our training set, we can train our classifier.
classifier = nltk.NaiveBayesClassifier.train(training_set)

#print(classifier.show_most_informative_features())
# contains(not) = False          positi : negati =      1.6 : 1.0
# contains(tired) = False        positi : negati =      1.2 : 1.0
# contains(excited) = False      negati : positi =      1.2 : 1.0
# contains(great) = False        negati : positi =      1.2 : 1.0
# contains(looking) = False      positi : negati =      1.2 : 1.0
# contains(like) = False         positi : negati =      1.2 : 1.0
# contains(love) = False         negati : positi =      1.2 : 1.0
# contains(amazing) = False      negati : positi =      1.2 : 1.0
# contains(enemy) = False        positi : negati =      1.2 : 1.0
# contains(about) = False        negati : positi =      1.2 : 1.0
# contains(best) = False         negati : positi =      1.2 : 1.0
# contains(forward) = False      positi : negati =      1.2 : 1.0
# contains(friend) = False       negati : positi =      1.2 : 1.0
# contains(horrible) = False     positi : negati =      1.2 : 1.0

pos_tweet = 'Larry is my friend'
#print(extract_features(tweet.split()))
# {'contains(amazing)': False, 'contains(morning)': False, 
# 'contains(love)': False, 'contains(great)': False, 
# 'contains(concert)': False, 'contains(not)': False, 
# 'contains(best)': False, 'contains(view)': False, 
# 'contains(excited)': False, 'contains(about)': False, 
# 'contains(horrible)': False, 'contains(enemy)': False, 
# 'contains(looking)': False, 'contains(like)': False, 
# 'contains(this)': False, 'contains(tired)': False, 
# 'contains(car)': False, 'contains(friend)': True, 
# 'contains(the)': False, 'contains(forward)': False, 
# 'contains(feel)': False}


senti_result = classifier.classify(extract_features(pos_tweet.split()))#positive
print(senti_result)

neg_tweet = "I do not like that man."
print(classifier.classify(extract_features(neg_tweet.split())))#negative

from collections import defaultdict
refsets = defaultdict(set)
testsets = defaultdict(set)
for i, (feats, label) in enumerate(test_tweets):
    refsets[label].add(i)
    observed = classifier.classify(extract_features(feats))
    testsets[observed].add(i)

testing_set = nltk.classify.apply_features(extract_features, test_tweets)
from nltk.metrics.scores import precision,recall,f_measure
print('pos precision:', precision(refsets['positive'], testsets['positive']))
print('pos recall:', recall(refsets['positive'], testsets['positive']))
print('pos F-measure:', f_measure(refsets['positive'], testsets['positive']))
print('neg precision:', precision(refsets['negative'], testsets['negative']))
print('neg recall:', recall(refsets['negative'], testsets['negative']))
print('neg F-measure:', f_measure(refsets['negative'], testsets['negative']))
print('accuracy ', nltk.classify.accuracy(classifier,testing_set)*100)




from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(MNB_classifier.classify(extract_features(pos_tweet.split())))#positive
print("Multinomial Algo accuracy percent:",nltk.classify.accuracy(MNB_classifier,testing_set)*100)

#not working for gaussian
# gaussian_classifier = SklearnClassifier(GaussianNB())
# gaussian_classifier.train(training_set)
# print("Gaussian Algo accuracy percent:",nltk.classify.accuracy(gaussian_classifier,testing_set)*100)

bernoulli_classifier = SklearnClassifier(BernoulliNB())
bernoulli_classifier.train(training_set)
print(bernoulli_classifier.classify(extract_features(pos_tweet.split())))#positive
print("Bernoulli Algo accuracy percent:",nltk.classify.accuracy(bernoulli_classifier,testing_set)*100)


from sklearn.linear_model import LogisticRegression, SGDClassifier

logistic_classifier = SklearnClassifier(LogisticRegression())
logistic_classifier.train(training_set)
print(logistic_classifier.classify(extract_features(pos_tweet.split())))#positive
print("Logistic Reg Algo accuracy percent:",nltk.classify.accuracy(logistic_classifier,testing_set)*100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print(SGD_classifier.classify(extract_features(pos_tweet.split())))#positive
print("SGD Algo accuracy percent:",nltk.classify.accuracy(SGD_classifier,testing_set)*100)

from sklearn.svm import SVC,LinearSVC,SVR, LinearSVR 
svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
print(svc_classifier.classify(extract_features(pos_tweet.split())))#positive
print("SVC Algo accuracy percent:",nltk.classify.accuracy(svc_classifier,testing_set)*100)

lsvc_classifier = SklearnClassifier(LinearSVC())
lsvc_classifier.train(training_set)
print(lsvc_classifier.classify(extract_features(pos_tweet.split())))#positive
print("LinearSVC Algo accuracy percent:",nltk.classify.accuracy(lsvc_classifier,testing_set)*100)


svr_classifier = SklearnClassifier(SVR())
svr_classifier.train(training_set)
print(svr_classifier.classify(extract_features(pos_tweet.split())))#positive
print("SVR Algo accuracy percent:",nltk.classify.accuracy(svr_classifier,testing_set)*100)

lsvr_classifier = SklearnClassifier(LinearSVR())
lsvr_classifier.train(training_set)
print(lsvr_classifier.classify(extract_features(pos_tweet.split())))#positive
print("LinearSVR Algo accuracy percent:",nltk.classify.accuracy(lsvr_classifier,testing_set)*100)

