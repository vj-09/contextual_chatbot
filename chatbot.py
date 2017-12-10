import tensorflow as tf
import tflearn
import pickle
import numpy as np
import json
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
intents = 'intents.json'
with  open(intents)as data:
    train_data = json.load(data)
set_data = train_data['intents']
#%%%%%%%%%%%        cleaning of data         %%%%%%%%%%%%%%%%%%#
documents = []
words = []
tags = []
responses = []
ignore_list = ['?',"'s"]
for data in set_data:
    for pattern in data['patterns']:
        w = nltk.word_tokenize(str(pattern))
        words.extend(w)
        documents.append((w,str(data['tag'])))
    tags.append(str(data['tag']))
    responses.extend(str(data['responses']))
words = [stemmer.stem(w.lower()) for w in words if w  not in ignore_list]
words = sorted(list(set(words)))
classes = list(set(tags))
#%%%%%%%%%%%%      Traing and out data           %%%%%%%%%#
training = []
output = []
output_temp = [0]*len(classes)
for doc in documents:
    bag = []
    doc_words = doc[0]
    doc_words =  [stemmer.stem(word.lower()) for word in doc_words]
    for word in words:
        if word in doc_words:
            bag.append(1)
        else:
            bag.append(0)
    class_list = [0]*len(classes)
    class_list[
    classes.index(doc[1])] = 1
    training.append([bag,class_list])
random.shuffle(training)
training = np.array(training)
print np.shape(training)
train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))
print np.shape(train_x),len(train_x[0])
#%%%%%%%%%%%%%% model2 with test %%%%%%%%%%%%%%%%%
input_features = len(train_x[0])
hidden_nodes1 = 10
hidden_nodes2 = 20
output_classes = len(train_y[0])
######## input layer to hidden1########
#27*47
#27*47 47*10 27*10
#27*10 10*20 27*20
#27*20 20*9 27*9
X = tf.placeholder(tf.float32,shape=[None,input_features],name='XXX')
Y = tf.placeholder(tf.float32,shape=[None,output_classes],name="YYY")
w1 = tf.Variable(tf.random_normal([input_features,hidden_nodes1]))
b1 = tf.Variable(tf.random_normal([hidden_nodes1]))
l1_out = tf.nn.softmax( tf.matmul(X,w1) + b1 )

########## hidden1 to hidden2########
w2 = tf.Variable(tf.random_normal([hidden_nodes1,hidden_nodes2]))
b2 = tf.Variable(tf.random_normal([hidden_nodes2]))
l2_out = tf.nn.softmax(tf.matmul(l1_out,w2) + b2)
############hidden2 to output ################

w3 =  tf.Variable(tf.random_normal([hidden_nodes2,output_classes]))
b3 = tf.Variable(tf.random_normal([output_classes]))
y = tf.nn.softmax(tf.matmul(l2_out,w3) + b3)
y_max = tf.argmax(y,1)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
epochs = 10000

for i in range(epochs):
    cross_ent_ , _ = sess.run([y_max,train_step], feed_dict={ X:train_x, Y:train_y })
    if i%100 == 0:
        print "training step : {}, cross_entropy : {}".format(i, cross_ent_)

################## test model ###########
def bow(sentance,words,show_details =False):
    # sentence_words = clean_input_sentance(sentence)
    sentence_words = nltk.word_tokenize(sentance)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    print words,sentence_words
    t = []
    for doc in sentence_words:
        bag = []
        # print doc
        for word in words:
            if word in doc:
                bag.append(1)
                print doc
            else:

                bag.append(0)
        t.append(bag)
    return t
#############response model###########
ERROR_THRESHOLD = 0.25
context = {}

def classify(sentence):
    return_list=[]
    return_list = bow(sentence,words)
    results = sess.run(y_max, feed_dict={ X:return_list})
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    print results
    # if we have a classification then find the matching intent tag
    if results:
         # loop as long as there are matches to process
        for a in results:
            for i in set_data:
            # find a tag matching the first result
                if i['tag'] == a[0]:
                    print i['tag'] , results[0][0]
                    if 'context_set' in i:
                        print ('context:',i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter']== context[userID]):
                            print ('tag:',i['tag'])
                            print random.choice(i['responses'])
            results.pop(0)
response("can we rent a mope")
response('today')
