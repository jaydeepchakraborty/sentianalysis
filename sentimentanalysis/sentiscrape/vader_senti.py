# not working
#http://dev.panlex.org/db/panlex_lite.zip
#pip3 install vaderSentiment
file = open("topics/marksafe/twitter_posts_init_train.txt", 'rt', encoding='latin1')
sentences = [file.read()]

        
import nltk
# nltk.download()
stoplist = []
stoplist = nltk.corpus.stopwords.words('english')
f = open('topics/marksafe/stopword','rt', encoding='latin1')
for word in f.read().split():
    stoplist.append(word)        

texts = [[word for word in sentences if word not in stoplist]]

testData = ['VADER is smart, handsome, and funny.',
            'VADER is smart, handsome, and funny!',
            'VADER is very smart, handsome, and funny.',
            'VADER is VERY SMART, handsome, and    FUNNY.',
            'VADER is VERY SMART, handsome, and FUNNY!!!',
            'VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!',
            'The book was good.',
            'The book was kind of good.',
            'The plot was good, but the characters are uncompelling and the dialog is not great.',
            'A really bad, horrible book.', "At least it isn't a horrible book.", ':) and :D',
            '',
            'Today sux',
            'Today sux!',
            'Today SUX!',
            "Today kinda sux! But I'll get by, lol"]


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for sentence in testData:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print("-------------------------------------")
