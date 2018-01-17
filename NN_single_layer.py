from numpy import exp, array, dot, random
import pandas as pd
import numpy as np

def data():
    train = pd.read_csv('/home/rik/train.csv')
    test = pd.read_csv('/home/rik/test.csv')
    y = train[['Survived']]
    x= train[['Age', 'Fare', 'Parch', 'SibSp', 'Embarked', 'Pclass', 'Sex']]
    x['Age'] = x['Age'].fillna(x['Age'].median())
    x['Embarked'][x['Embarked']=='C']=0.0
    x['Embarked'][x['Embarked']=='Q']=0.5
    x['Embarked'][x['Embarked']=='S']=1.0
    x['Sex'][x['Sex']=='male']=0.0
    x['Sex'][x['Sex']=='female']=1.0
    x['Embarked'] = x['Embarked'].fillna(x['Embarked'].median())
    xt = test[['Age', 'Fare', 'Parch', 'SibSp', 'Embarked', 'Pclass', 'Sex']]
    xt['Age'] = xt['Age'].fillna(xt['Age'].median())
    xt['Fare'] = xt['Fare'].fillna(xt['Fare'].median())
    xt['Embarked'][xt['Embarked']=='C']=0.0
    xt['Embarked'][xt['Embarked']=='Q']=0.5
    xt['Embarked'][xt['Embarked']=='S']=1.0
    xt['Sex'][xt['Sex']=='male']=0.0
    xt['Sex'][xt['Sex']=='female']=1.0
    x['Age'] = (x['Age'] - x['Age'].min())/(x['Age'].max() -x['Age'].min())
    x['Fare'] = (x['Fare'] - x['Fare'].min())/(x['Fare'].max() -x['Fare'].min())
    x['Parch'] = (x['Parch'] - x['Parch'].min())/(x['Parch'].max() -x['Parch'].min())
    x['SibSp'] = (x['SibSp'] - x['SibSp'].min())/(x['SibSp'].max() -x['SibSp'].min())
    x['Pclass'] = (x['Pclass'] - x['Pclass'].min())/(x['Pclass'].max() -x['Pclass'].min())


    xt['Age'] = (xt['Age'] - xt['Age'].min())/(xt['Age'].max() -xt['Age'].min())
    xt['Fare'] = (xt['Fare'] - xt['Fare'].min())/(xt['Fare'].max() -xt['Fare'].min())
    xt['Parch'] = (xt['Parch'] - xt['Parch'].min())/(xt['Parch'].max() -xt['Parch'].min())
    xt['SibSp'] = (xt['SibSp'] - xt['SibSp'].min())/(xt['SibSp'].max() -xt['SibSp'].min())
    xt['Pclass'] = (xt['Pclass'] - xt['Pclass'].min())/(xt['Pclass'].max() -xt['Pclass'].min())
    
    x = np.array(x.values, dtype=np.float)
    y = np.array(y.values, dtype=np.float)
    xt = np.array(xt.values, dtype=np.float)
    return (x,y,xt)

class NN():
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((7,1))-1
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def train(self, train_x, train_y, iterations):
        for i in range(iterations):
            output = self.think(train_x)
            error = train_y - output
            adjustment = dot(train_x.T, error * self.sigmoid_derivative(output))
            self.weights += adjustment

    def think(self, inputs):
        return self.sigmoid(dot(inputs,self.weights))

if __name__ == '__main__':
    neural_network = NN()
    print('original weights are:')
    print(neural_network.weights)
    training_set_inputs, training_set_outputs, test = data()
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    print('New weights are:')
    print(neural_network.weights)
    res = neural_network.think(test)
    for i in range(len(res)):
        if res[i][0] < 0.5:
            res[i][0] = 0
        else:
            res[i][0] = 1
    out = pd.read_csv('/home/rik/test.csv', sep = ',', usecols = ['PassengerId'])
    out['Survived'] = res
    out['Survived'] = out['Survived'].astype(int)
    out.to_csv('newsub.csv', index = False)
