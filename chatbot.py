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
# print set_data
########################################################################
#%%%%%%%%%%%        cleaning of data         %%%%%%%%%%%%%%%%%%#
documents = []
words = []
tags = []
responses = []
ignore_list = ['?',"'s"]
for data in set_data:
    # print data['patterns']
    for pattern in data['patterns']:
        # print pattern
        w = nltk.word_tokenize(str(pattern))
        # print w
        words.extend(w)
        documents.append((w,str(data['tag'])))
    tags.append(str(data['tag']))
    responses.extend(str(data['responses']))
words = [stemmer.stem(w.lower()) for w in words if w  not in ignore_list]
# words.append('accept')
words = sorted(list(set(words)))
classes = list(set(tags))
    # print words
# print (len(documents), "documents")
# print (len(classes), "classes", classes)
# print (len(words), "unique stemmed words", words)
###############################################################################
#%%%%%%%%%%%%      Traing and out data           %%%%%%%%%#
training = []
output = []
output_temp = [0]*len(classes)
# print output_temp
for doc in documents:
    bag = []
    doc_words = doc[0]
    doc_words =  [stemmer.stem(word.lower()) for word in doc_words]
    # print doc_words
    for word in words:
        if word in doc_words:
            bag.append(1)
        else:
            bag.append(0)
    # print bag
    class_list = [0]*len(classes)
    class_list[
    classes.index(doc[1])] = 1
    # print class_list
    training.append([bag,class_list])
random.shuffle(training)
training = np.array(training)
print np.shape(training)
train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))
print np.shape(train_x),len(train_x[0])
###########################################################################
#%%%%%%%%%%% tensorflow veriables and placeholders dec %%%%%%%%%%%%%%%%%#
tf.reset_default_graph()
net = tflearn.input_data(shape=[None,len(train_x[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=800, batch_size=8, show_metric=True)

print model.save('model.tflearn')
###################################################################################
#%%%%%%%%%%%%%% model2 with test %%%%%%%%%%%%%%%%%
# input_features = len(train_x[0])
# hidden_nodes1 = 10
# hidden_nodes2 = 20
# output_classes = len(train_y[0])
# ######## input layer to hidden1########
# #27*47
# #27*47 47*10 27*10
# #27*10 10*20 27*20
# #27*20 20*9 27*9
# X = tf.placeholder(tf.float32,shape=[None,input_features],name='XXX')
# Y = tf.placeholder(tf.float32,shape=[None,output_classes],name="YYY")
# w1 = tf.Variable(tf.random_normal([input_features,hidden_nodes1]))
# b1 = tf.Variable(tf.random_normal([hidden_nodes1]))
# l1_out = tf.nn.softmax( tf.matmul(X,w1) + b1 )
#
# ########## hidden1 to hidden2########
# w2 = tf.Variable(tf.random_normal([hidden_nodes1,hidden_nodes2]))
# b2 = tf.Variable(tf.random_normal([hidden_nodes2]))
# l2_out = tf.nn.softmax(tf.matmul(l1_out,w2) + b2)
# ############hidden2 to output ################
#
# w3 =  tf.Variable(tf.random_normal([hidden_nodes2,output_classes]))
# b3 = tf.Variable(tf.random_normal([output_classes]))
# y = tf.nn.softmax(tf.matmul(l2_out,w3) + b3)
# y_max = tf.argmax(y,1)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
#
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# epochs = 20000
#
# for i in range(epochs):
#     # print train_x[10],train_y[0]
#     # print sess.run(train_step, feed_dict={ X:train_x, Y:train_y })
#     cross_ent_ , _ = sess.run([y_max,train_step], feed_dict={ X:train_x, Y:train_y })
#     if i%100 == 0:
#         print "training step : {}, cross_entropy : {}".format(i, cross_ent_)


########### test model2 ###################
# with  open("test_data.json")as data:
    # test_data = json.load(data)
# set_data = ['work']
# test_document = []
# for data in set_data:
#     # print data['patterns']
#     # print pattern
#     w1 = nltk.word_tokenize(str(data))
#     test_words = [stemmer.stem(w.lower()) for w in w1 if w  not in ignore_list]
#     test_document.append(test_words)
# test_x = []
# for doc in test_document:
#     bag = []
#     for word in words:
#         if word in doc:
#             bag.append(1)
#         else:
#             bag.append(0)
#     test_x.append(bag)
# print test_document
# print set_data
# out = sess.run(y_max, feed_dict={ X:test_x })
# print classes
# for i in range(len(out)):
#     print out[i]
#     print "data : {} class : [ {} ]".format(set_data[i], classes[out[i]])
# pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

################### test model ###########
def bow(sentance,words,show_details =False):
    # sentence_words = clean_input_sentance(sentence)
    sentence_words = nltk.word_tokenize(sentance)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i] = 1
                if show_details:
                    print ("found%s"%w)
    return np.array(bag)
p = bow("is your shop open today",words)


#############response model###########
ERROR_THRESHOLD = 0.25
context = {}

def classify(sentence):
    return_list=[]
    results = model.predict([bow(sentence,words)])[0]
    # print results
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # print results
    results.sort(key=lambda x: x[1], reverse=True)
    # print results
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # print return_list
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)

    # if we have a classification then find the matching intent tag
    if results:

        # loop as long as there are matches to 0process
        for a in results:
            print a
            for i in set_data:
                # print i
                # find a tag matching the first result
                if i['tag'] == a[0]:
                    # print i
                    # print i['tag'] , results[0][0]
                    # a random response from the intent
                    # print sentence, i['responses']
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
