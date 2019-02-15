#Matt Fein
#Logistic Regression
#MachineLearing


import numpy as np
from numpy import genfromtxt

class LogRegress:

    def __init__(self, learnRate = .01, iterations = 100000, fitInt=True, verbose=False):
        self.learnRate = learnRate
        self.iterations = iterations
        self.fitInt = fitInt
        self.verbose = verbose

    # Calculate sig
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    # Calculate Loss
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


    def fit(self, X, y):
        if self.fitInt:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[::])
        X.shape[::]

        for i in range(self.iterations):
            z = np.vdot(X, self.theta)
            h = self.sigmoid(z)
            grad = np.dot(X.T, (h - y)) / y.size
            self.theta = self.theta - self.learnRate * grad

            if (self.verbose == True and i % 100 == 0):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
            

    def predict_probability(self, X):
        if self.fitInt:
            X = self.add_intercept(X)
        print(self.sigmoid(np.vdot(X, self.theta)))
        return self.sigmoid(np.vdot(X, self.theta))


    def predict(self, X):
        return self.predict_probability(X).round()



#StufftoRun
data = genfromtxt('SPECT.train', delimiter=',')
test = genfromtxt('SPECT.test', delimiter=',')
yDiagnosis = data[:,0]
xFeatures = data[:, 1::]
xpredictFeatures = test[:, 1::]
ypredictDiagnosis = test[:,0]
#Train
model = LogRegress(learnRate=0.1, iterations=300000)
model.fit(xFeatures, yDiagnosis)
preds = model.predict(xpredictFeatures)
print('Accuracy of Model: ', (preds == yDiagnosis).mean())
print('')
# print(model.theta)
# accuracy

