# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:48:47 2020

@author: Charlène BRUNO & Loïc CHEONG
"""

import numpy as np
import pandas as pd
import math as m

#get data from text file
messages = pd.read_csv('messages.txt', sep='\t')
messages.columns= ["nature", "message"] #Renaming the 2 columns

print(messages)
print(type(messages))
messages["nature"] = messages["nature"].map({'spam':1, 'ham':0})


#Training set
messages_copy = messages.copy()
train_set = messages_copy.sample(frac=0.75, random_state=0) #training _set represents 75% of the original dataset
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


#Creation of dictionnary
def makeDictionnary(messagesArray):
    """
Input param : messageArray => A Series with a string message in each line
Output param : dictionary => A list of words extracted from messagesArray
    A word appears only once in the dictionary in minuscule letter and have more than 2 letters
    The meaning of the words are not processed here
    """
    
    #Initialize an empty list
    dictionary=[] 
    
    for i in messagesArray.index:
        #print(messagesArray[i])
        words = messagesArray[i].split(' ') #Turns the string messageArray[i] into a list of strings (words)
#        print(words)
        for word in words:
            if(len(word)>2 and word.isalpha() == True): 
                #Exclusion des mots moins de 2 lettres et avec des caractères non-aphabétiques
                if  word.lower() not in dictionary:
                    #Add the word in minuscule letter in the dictionary in order to avoid to compare the same in CAPITAL and minuscule letter
                    dictionary.append(word.lower()) 
    
    #RECODER POUR AJOUTER LES MOTS AVEC DES VIRGULES, SINON CELA FAUSSERA LE CALCUL DE PROBABILITE (possibilité qu'une proba soit égale à zéro)
    
    #print(dictionary)
    return dictionary


#Extraction of features
def extract_features(dictionary, messagesArray):
    """
Input param :   messageArray => A Series with a string message in each line
                dictionary => A list of words extracted from messagesArray
Output param :  features_matrix => An array
        The function extract_features returns an array of zeros and ones
        The row of extract_features represents le list of words of a message
        The column is the word (example: column 0 is the word "opps", extract_features[0][0] = 1, means that the word is present
        Otherwise, it's not present)
    """
    
    #Initalization
    features_matrix = np.zeros((len(messagesArray), len(dictionary)))
    docID = 0 #	message number/index in messagesArray
    
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
    # X and Y are both DataFrame
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
        Y.index = [i for i in range(I)] #Rename the index of Y by [0,1,2,3,...,I-1,I]
        
        #Compute Phi_y_MLE = P(Y=1)
        Phi_y_MLE = Y.sum()/I
        
        #1st loop aims to fill the ith line of y_predict
        for i in range(I):
            #2nd loop aims to fill the yth column of y_predict indoer to compute P(x|y)*P(y) for a given y ={0,1}
            message = [wordIndex for wordIndex in range(N) if features_matrix[i][wordIndex]>0 ]
            n = len(message)
#            print(i, message)
            for y in range(2):
                Sum_ouside_1st_log = 0
                
                for wordIndex in range(n):
                    
                    #Start computing phi_n|y_MLE which is the variable "Sum"
                    Sum_inside_1st_log = 0 #variable to store log(sum of indicator of xn(i)|y)
                    for line in range(I):
                        
                        if (features_matrix[line][wordIndex]>0 and Y[line]==y):
                            Sum_inside_1st_log += features_matrix[line][wordIndex]
#                    print(word, Sum_inside_1st_log)
                    if (Sum_inside_1st_log>0):
                        Sum_ouside_1st_log = Sum_ouside_1st_log + m.log(Sum_inside_1st_log)
                    
                if (y==0):
                    y_predict[i][y] = Sum_ouside_1st_log - n*m.log(I-Y.sum()) + m.log((1-Phi_y_MLE)) #P(x|y=0)*P(y=0)
                else: 
                    y_predict[i][y] = Sum_ouside_1st_log - n*m.log(Y.sum()) + m.log(Phi_y_MLE) #P(x|y=1)*P(y=1)
            print(i, y_predict[i])
#        print(y_predict) #print both y* for y=0 and y=1
        
        #Return the index of the max value of the line i  
        #Dim(y_predict)=(1,I)
        y_predict = np.argmax(y_predict, axis=1) # y*
        
        return y_predict


#print("Starting NaiveBayes programm for the training sets")
#print("\ty=0\t\ty=1")
#y_predict_train = NaiveBayes(X_training, Y_training)
print("Starting NaiveBayes programm for the test sets")
print("\ty=0\t\ty=1")
y_predict_test = NaiveBayes(X_test, Y_test)
##
#df = pd.DataFrame(data=features_matrix_train,columns=dico_train)
#Y_training.index = [i for i in range(3748)]
#df.insert(0, "Y", Y_training, True) 
#for word in dico_train:
#    Y = df[(df[word]>0) & (df["Y"]==1)]
#for word in dico_train:
    

