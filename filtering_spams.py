# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:48:47 2020

@author: Charlène BRUNO & Loïc CHEONG
"""

import numpy as np
import pandas as pd

#get data from text file
messages = pd.read_csv('messages.txt', sep='\t')
messages.columns= ["nature", "message"]

print(messages)
print(type(messages))
messages["nature"] = messages["nature"].map({'spam':1, 'ham':0})


#Training set
messages_copy = messages.copy()
train_set = messages_copy.sample(frac=0.75, random_state=0)
print ('Training set')
X_training = train_set["message"]
Y_training = train_set["nature"]
#print(X_training)
print("Number of spams : ",Y_training.sum(),"/",len(Y_training)," (",round(Y_training.sum()/len(Y_training)*100,2),"%)\n")


#Test set
test_set = messages_copy.drop(train_set.index)
print ('Test set')
X_test = test_set["message"]
Y_test = test_set["nature"]
print("Number of spams : ",Y_test.sum(),"/",len(Y_test)," (",round(Y_test.sum()/len(Y_test)*100,2),"%)\n")
#print(X_test)
#print(Y_test)


#creation of dictionnary
def makeDictionnary(messagesArray):
    dictionary=[] #Init an empty list
    for i in messagesArray.index:
        #print(messagesArray[i])
        words = messagesArray[i].split(' ')
#        print(words)
        for k in words:
            if(len(k)>2 and k.isalpha() == True): 
                #Exclusion des mots moins de 2 lettres et avec des caractères non-aphabétiques
                if  k.lower() not in dictionary:
                    dictionary.append(k.lower()) #Add the word in minuscule letter in the dictionary
    
    #RECODER POUR AJOUTER LES MOTS AVEC DES VIRGULES, SINON CELA FAUSSERA LE CALCUL DE PROBABILITE (possibilité qu'une proba soit égale à zéro)
    
    #print(dictionary)
    return dictionary


#extraction of features
def extract_features(dictionary, messagesArray):
    #files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(messagesArray), len(dictionary)))
    docID = 0 #	message number/index
    for i in messagesArray.index:
        words = messagesArray[i].split(' ')
        words = [x.lower() for x in words] 
        for word in words:
#            if word=="oops" : 
#               print(word)
            if word in dictionary:
#                if word=="oops" : print("\t ok '",docID," \n")
                features_matrix[docID, dictionary.index(word)] = words.count(word) 
        docID = docID + 1
    return features_matrix


#creation dico
dico_train = makeDictionnary(X_training)
features_matrix_train = extract_features(dico_train, X_training)



#Implementation of Naive Bayes programm
def NaiveBayes(X,Y):
    #X is the X_training or X_test
    #Y is the Y_training or Y_test
    if len(X)!=len(Y):
        #In order to avoid X_test and Y_train as input data
        return "LENGTH ERROR IN THE INPUT DATA !"
    else:
#        I = len(Y)
        dico = makeDictionnary(X)
        features_matrix = extract_features(dico, X)
        (I, N) = features_matrix.shape ##(nb of msg, nb of words)
        y_predict = np.zeros((I,2)) 
        Y.index = [i for i in range(I)] #[0,1,2,3,...,I-1,I]
        
        #Compute Phi_y_MLE = P(Y=1)
        Phi_y_MLE = Y.sum()/I
        
        #1st loop aims to fill the ith line of y_predict
        for i in range(I):
            
            #2nd loop aims to fill the yth column of y_predict indoer to compute P(x|y)*P(y) for a given y ={0,1}
            for y in range(2):
                Product = 1
                for word in range(N):
                    
                    #Start computing phi_n|y_MLE which is the variable "Sum"
                    Sum = 0 #somme sur le numérateur de phi_n|y_MLE
                    for line in range(I):
                        if (features_matrix[line][word]>0 and Y[line]==y):
                            Sum += features_matrix[line][word]
                    if (y==0):
                        Sum = Sum/(I-Y.sum()) #phi_n|y0_MLE
                    else:#y=1
                        Sum = Sum/Y.sum() #phi_n|y1_MLE
                    
                    Product = Product * Sum #On multiplie tous les phi_n|y_MLE pour un y donné #PHI_Total = PHI(n-1)*PHI(n)
                if (y==0):
                    y_predict[i][y] = Product*(1-Phi_y_MLE) #P(x|y=0)*P(y=0)
                else: 
                    y_predict[i][y] = Product*Phi_y_MLE #P(x|y=1)*P(y=1)
            print(i, y_predict[i])
        print(y_predict) #print both y* for y=0 and y=1
        
        #Return the index of the max value of the line i  
        #Dim(y_predict)=(1,I)
        y_predict = np.argmax(y_predict, axis=1) # y*
        
        return y_predict


print("Starting NaiveBayes programm for the training sets")
NaiveBayes(X_training, Y_training)

#
#df = pd.DataFrame(data=features_matrix_train,columns=dico_train)
#Y_training.index = [i for i in range(3748)]
#df.insert(0, "Y", Y_training, True) 
#for word in dico_train:
#    Y = df[(df[word]>0) & (df["Y"]==1)]
#for word in dico_train:
    

