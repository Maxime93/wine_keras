import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
# from keras.datasets import boston_housing, mnist, cifar10, imdb

# (x_train,y_train),(x_test,y_test) = mnist.load_data()
# (x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
# (x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
# (x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
# num_classes = 10

from urllib2 import urlopen
# data = np.loadtxt(urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),delimiter=",")
# X = data[:,0:8]
# y = data [:,8]


data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))

print data.shape
print labels.shape

model = Sequential()
print model.summary()

# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(data,labels,epochs=10,batch_size=32)
# predictions = model.predict(data)