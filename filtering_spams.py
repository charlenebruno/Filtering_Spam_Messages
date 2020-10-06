# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:48:47 2020

@author: Charlène BRUNO & Loïc CHEONG
"""

import numpy as np
import pandas as pd

#get data from text file
messages = pd.read_csv('messages.txt', sep='\t')
#messages.str.upper()
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
print ('Training set')
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
    docID = 0
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
    if len(X)!=len(Y):
        #In order to avzoid X_test and Y_train as input data
        return "LENGTH ERROR OF THE INPUT DATA !"
    else:
#        I = len(Y)
        dico = makeDictionnary(X)
        features_matrix = extract_features(dico, X)
        (I, N) = features_matrix.shape
        y_predict = np.zeros((I,2)) 
        Y.index = [i for i in range(I)]
        
        #Compute Phi_y_MLE = P(Y=1)
        Phi_y_MLE = Y.sum()/I
        
        for i in range(I):
            for j in range(2):
                Product = 1
                for word in range(N):
                    Sum = 0 #somme sur le numérateur de phi_n|y_MLE
                    for line in range(I):
                        if (features_matrix[line][word]>0 and Y[line]==j):
                            Sum += features_matrix[line][word]
                    if (j==0):
                        Sum = Sum/(I-Y.sum()) #phi_n|y0_MLE
                    else:
                        Sum = Sum/Y.sum() #phi_n|y1_MLE
                    Product *= Sum #On multiplie tous les phi_n|y_MLE
                if (j==0):
                    y_predict[i][j] = Product*(1-Phi_y_MLE)
                else: 
                    y_predict[i][j] = Product*Phi_y_MLE
        print(y_predict)
        y_predict = np.argmax(y_predict, axis=1) #Return the index of the max value of the line i. Dim(y_predict)=(1,I)
        return y_predict


#Remplacer Les index de y_train et y_test
#Y_training.index = [i for i in range(len(Y_training))]
#Y_test.index = [i for i in range(len(Y_test))]

NaiveBayes(X_training, Y_training)
