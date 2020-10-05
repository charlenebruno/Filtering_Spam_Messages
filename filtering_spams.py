import pandas as pd

#get data from text file
messages = pd.read_csv(
'messages.txt', sep='\t')
messages.columns= ["nature", "message"]
print(messages)
print(type(messages))
messages["nature"] = messages["nature"].map({'spam':1, 'ham':0})

#training set
messages_copy = messages.copy()
train_set = messages_copy.sample(frac=0.75, random_state=0)
print ('Training set')


X_training = train_set["message"]
Y_training = train_set["nature"]
print(X_training)
#print(Y_training)

#test set
test_set = messages_copy.drop(train_set.index)

X_test = test_set["message"]

Y_test = test_set["nature"]
#print(X_test)
#print(Y_test)

#creation of dictionnary
def makeDictionnary(messageArray):
    L=[] #Initialisation d'un liste vide
    for i in messageArray.index:
        #print(messageArray[i])
        words = messageArray[i].split(' ')

        #print(words)
        for k in words:
            L.append(k)
    print(L)

makeDictionnary(X_training)


