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
def makeDictionnary(messagesArray):
    dictionnary=[] #Initialisation d'un liste vide
    for i in messagesArray.index:
        #print(messagesArray[i])
        words = messagesArray[i].split(' ')

        #print(words)
        for k in words:
            if(len(k)>2 and k.isalpha() == True):
                dictionnary.append(k)
    print(dictionnary)
    return dictionnary

#extraction of features
def extract_features(dictionnary, X_training, X_test):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
                                docID = docID + 1
    return features_matrix

#creation dico
dico = makeDictionnary(X_training)
#features_matrix = extract_features(dico, X_training, X_test)



