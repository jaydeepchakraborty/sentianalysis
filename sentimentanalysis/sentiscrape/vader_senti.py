#pip3 install vaderSentiment
file = open("twitter_posts.txt", 'rt', encoding='latin1')
sentences = [file.read()]

        
import nltk
stoplist = []
stoplist = nltk.corpus.stopwords.words('english')
f = open('stopword','rt', encoding='latin1')
for word in f.read().split():
    stoplist.append(word)        

texts = [[word for word in sentences if word not in stoplist]]

testData = ['VADER is smart, handsome, and funny.', 'VADER is smart, handsome, and funny!', 'VADER is very smart, handsome, and funny.', 'VADER is VERY SMART, handsome, and    FUNNY.', 'VADER is VERY SMART, handsome, and FUNNY!!!', 'VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!', 'The book was good.', 'The book was kind of good.',    'The plot was good, but the characters are uncompelling and the dialog is not great.', 'A really bad, horrible book.', "At least it isn't a horrible book.", ':) and :D', '',   'Today sux', 'Today sux!', 'Today SUX!', "Today kinda sux! But I'll get by, lol"]


from vaderSentiment.vaderSentiment import sentiment
for sentence in testData:
        print(sentence)
        vs = sentiment(sentence)
        print("\n\t" + str(vs))

