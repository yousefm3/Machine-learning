import sys
import numpy as np

class KNN:

    def __init__(self, train_x, train_y, k):
        self.k = k
        self.X_train = train_x
        self.y_train = train_y


    def predictions(self, test_x):
        predicted_labels = [self.train(x) for x in test_x]
        return np.array(predicted_labels)

    def train(self, x):
        # calcuate the distances
        distances = [np.sqrt(np.sum((x-x_train)**2)) for x_train in self.X_train]
        # Pick the first K entries from the sorted collection 
        k_indices = np.argsort(distances)[: self.k]
        k_list = [self.y_train[i] for i in k_indices]
        mode_num = max(set(k_list), key = k_list.count)
        return int(mode_num)


class PERC:
    def __init__(self, train_x, train_y, lr):
        self.xTrain = train_x
        self.yTrain = train_y
        self.lr = lr
    
    def train(self, epochs):
        bias = np.full(240,1).reshape(240,1)
        self.xTrain = np.append(self.xTrain, bias, axis=1)
        w = np.zeros((3, len(self.xTrain[0])), dtype=float)
        dataSet = list(zip(self.xTrain, self.yTrain))
        np.random.shuffle(dataSet)
        for i in range(epochs):
           for x, y in dataSet:
               y_hat = np.argmax(np.dot(w, x))
               if y_hat != y:
                   w[int(y)] = w[int(y)] + (self.lr * x)
                   w[y_hat] = w[y_hat] - (self.lr * x)
           self.lr *= 0.5

        return w
      
    def predictions(self,w,normalized_test):
        predictions = []
        for normalized_row in zip(normalized_test):
           row = np.append(normalized_row, 1)
           pr_yh = np.argmax(np.dot(w, row))
           predictions.append(pr_yh)
        return predictions
        

class SVM:
    def __init__(self, train_x, train_y, lr, lamda):
        self.xTrain = train_x
        self.yTrain = train_y
        self.lr = lr
        self.lamda = lamda
    
    def train(self, epochs):
        bias = np.full(240,1).reshape(240,1)
        self.xTrain = np.append(self.xTrain, bias, axis=1)
        w = np.zeros((3, len(self.xTrain[0])), dtype=float)
        dataSet= list(zip(self.xTrain, self.yTrain))
        np.random.shuffle(dataSet)
        for e in range(epochs):
           for x, y in dataSet:
               yh = np.argmax(np.dot(w, x))
               if (y != yh):
                   if y != yh:
                       y = int(y)
                       yh = int(yh)
                       w[y] = (1 - self.lr * self.lamda) * w[y] + self.lr * x
                       w[yh] = (1 - self.lr * self.lamda) * w[yh] - self.lr * x
                   for i in range(w.shape[0]):
                       if i != y and i != yh:
                           w[i] = (1 - self.lr * self.lamda) * w[i]
           self.lr *= 0.5
        return w
    
    def predictions(self,w,normalized_test):
        predictions = []
        for normalized_row in zip(normalized_test):
           row = np.append(normalized_row, 1)
           pr_yh = np.argmax(np.dot(w, row))
           predictions.append(pr_yh)
        return predictions
    
#Passive Aggressive
class PA:
    def __init__(self, train_x, train_y):
        self.xTrain = train_x
        self.yTrain = train_y
    
    def train(self, epochs):
        bias = np.full(240,1).reshape(240,1)
        self.xTrain = np.append(self.xTrain, bias, axis=1)
        w = np.zeros((3, len(self.xTrain[0])), dtype=float)
        for ep in range(epochs):
            dataSet = list(zip(self.xTrain, self.yTrain))
            np.random.shuffle(dataSet)
            for x, y in dataSet:
                yh = np.argmax(np.dot(w, x))
                if y != yh:
                  y = int(y)
                  yh = int(yh)
                  loss = max(0, 1 - np.dot(w[y],x) + np.dot(w[yh], x))
                  tau = loss / (2 * (np.linalg.norm(x)**2))
                  w[y] = w[y] + tau * x
                  w[yh] = w[yh] - tau * x
        return w
    
    def predictions(self,w,normalized_test):
        predictions = []
        for normalized_row in zip(normalized_test):
           row = np.append(normalized_row, 1)
           pa_yh = np.argmax(np.dot(w, row))
           predictions.append(pa_yh)
        return predictions

def normalize(data_set):
    return (data_set - data_set.mean(0)) / data_set.std(0)


def main():
    train_x = np.loadtxt(sys.argv[1],delimiter = ',')
    train_y = np.loadtxt(sys.argv[2])
    test_x = np.loadtxt(sys.argv[3],delimiter = ',')
    f = open(sys.argv[4],"w")

    normalized_train = normalize(train_x)
    normalized_test = normalize(test_x)
    
    #knn
    knn = KNN(train_x, train_y, 5)
    knn_predictions = knn.predictions(test_x)
    
    #SVM
    svm = SVM(normalized_train,train_y,0.1,0.1)
    w = svm.train(50)
    svm_predictions = svm.predictions(w, normalized_test)
    
    #perceptron
    perc = PERC(normalized_train,train_y,0.1)
    w = perc.train(50)
    perc_predictions = perc.predictions(w, normalized_test)
    
    #PA
    pa = PA(normalized_train,train_y)
    w = pa.train(85)
    pa_predictions = pa.predictions(w, normalized_test)
    #print(metrics.accuracy_score(knn_predictions, train_y))
    for knn_yhat,perceptron_yhat, svm_yhat, pa_yhat, in zip(knn_predictions,perc_predictions, svm_predictions, pa_predictions):
        f.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")
    f.close()

if __name__ == '__main__':
    main()
