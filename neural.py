import pandas
from sklearn.neural_network import MLPClassifier

def preprocess(data):
    data=data.fillna(0)
    data['Sex']=data['Sex'].replace('male',1)
    data['Sex']=data['Sex'].replace('female',0)
    data['Embarked']=data['Embarked'].replace('S',1)
    data['Embarked']=data['Embarked'].replace('Q',2)
    data['Embarked']=data['Embarked'].replace('C',3)
    return data

features=['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']
train_data = pandas.read_csv('train/train.tsv', header=0, sep='\t')
X_train=train_data[features]
X_train=preprocess(X_train)
Y_train=train_data['Survived']

columns=['PassengerId','Pclass','Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare','Cabin','Embarked']
X_dev= pandas.read_csv('dev-0/in.tsv', header=None, sep='\t', names=columns)
X_dev=X_dev[features]
X_dev=preprocess(X_dev)

X_test= pandas.read_csv('test-A/in.tsv', header=None, sep='\t', names=columns)
X_test=X_test[features]
X_test=preprocess(X_test)

model=MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=5)
model.fit(X_train,Y_train)

predicted_ydev = model.predict(X_dev)
#print(predicted_ydev[0:20])
pandas.DataFrame(predicted_ydev).to_csv('dev-0/out.tsv', index=None, header=None, sep='\t')

predicted_ytest = model.predict(X_test)
#print(predicted_ydev[0:20])
pandas.DataFrame(predicted_ytest).to_csv('test-A/out.tsv', index=None, header=None, sep='\t')

