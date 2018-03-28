
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 4: Classification with QDA, LDA, and Logistic Regression

# Damian Armijo

# ## Overview

# In this assignment, you will make a new version of your ```NeuralNetwork``` class called ```NeuralNetworkClassifier```. You will then apply ```QDA```, ```LDA``` and your ```NeuralNetworkClassifier``` to a classification problem and discuss the results.  The ```tanh``` function will be used as the activation function for ```NeuralNetworkClassifier```.

# ### NeuralNetworkClassifier

# Copy your ```neuralnetworksA2.py``` into a new file named ```neuralnetworksA4.py```.  Define a new class named ```NeuralNetworkClassifier``` that extends ```NeuralNetwork```.  The following code cell indicates which methods you must override, with comments instructing you what you must do to complete it.  Add this class to your ```neuralnetworksA4.py``` file.

# ### Test It

# In[1]:


import numpy as np
import neuralnetworksA4 as nn


# In[2]:


X = np.arange(10).reshape((-1, 1))
T = np.array([1]*5 + [2]*5)


# In[3]:


netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(X, T, 20)
print(netc)


# In[4]:



print('T, Predicted')
#print(T)
#print(netc.use(X))
print(np.hstack((T.reshape(-1,1), netc.use(X))))


# ## Addition to ```partition``` function

# Add the keyword parameter ```classification``` with a default value of ```False``` to your ```partition``` function.  When its value is set to ```True``` your ```partition``` function must perform a stratified partitioning as illustrated in lecture notes [12 Introduction to Classification](http://nbviewer.ipython.org/url/www.cs.colostate.edu/~anderson/cs445/notebooks/12%20Introduction%20to%20Classification.ipynb)     

# In[5]:


print(X)
print(T)


# In[5]:


import mlutilities as ml
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T.reshape(-1,1), 0.6, classification=True)


# In[ ]:


Xtrain


# In[ ]:


Ttrain


# In[ ]:


Xtest


# In[ ]:


Ttest


# ### Example with toy data

# Use the above data to compare QDA, LDA, and linear and nonlinear logistic regression.

# In[6]:


import qdalda
qda = qdalda.QDA()
qda.train(Xtrain, Ttrain)
Ytrain = qda.use(Xtrain)
Ytest = qda.use(Xtest)


# In[7]:


print(np.hstack((Ttrain, Ytrain)))


# In[8]:


np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[9]:


print(np.hstack((Ttest, Ytest)))


# In[10]:


np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[11]:


lda = qdalda.LDA()
lda.train(Xtrain, Ttrain)
Ytrain = lda.use(Xtrain)
Ytest = lda.use(Xtest)


# In[12]:


print(np.hstack((Ttrain, Ytrain)))


# In[13]:


print(np.hstack((Ttest, Ytest)))


# In[14]:


np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[15]:


np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[16]:


ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[17]:


ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# In[34]:


netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[35]:


print((Ttrain.shape,Ytrain.reshape(-1,1).shape))
print(np.hstack((Ttrain, Ytrain.reshape(-1,1))))


# In[36]:


print(np.hstack((Ttest, Ytest.reshape(-1,1))))


# In[37]:


np.sum(Ttrain == Ytrain.reshape(-1,1)) / len(Ttrain) * 100


# In[38]:


np.sum(Ttest == Ytest.reshape(-1,1)) / len(Ttest) * 100


# In[23]:


ml.confusionMatrix(Ttrain, Ytrain.reshape(-1,1), [1, 2]);


# In[24]:


ml.confusionMatrix(Ttest, Ytest.reshape(-1,1), [1, 2]);


# Remember that linear logistic regression can be applied by specifying 0 hidden units.

# In[25]:


netc = nn.NeuralNetworkClassifier(X.shape[1], 0, len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[26]:


ml.confusionMatrix(Ttrain, Ytrain.reshape(-1,1), [1, 2]);


# In[27]:


ml.confusionMatrix(Ttest, Ytest.reshape(-1,1), [1, 2]);


# ## Apply to data from orthopedic patients

# Download ```column_3C_weka.csv``` from [this Kaggle site](https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients).  Use the column named ```class``` to create your target class labels. Apply QDA, LDA, linear logistic regression, and nonlinear logistic regression to this data.  Experiment with different hidden layer structures and numbers of iterations and describe what you find.
# 
# Partition data into 80% for training and 20% for testing, with ```shuffle=True```.
# 
# Print percents of training and testing samples correctly classified by QDA, LDA and various neural network classifiers.  Also print confusion matrices for training and for testing samples for each classifier.  Discuss the relative performance of your classifiers.

# In[2]:


import pandas
import numpy as np
import mlutilities as ml

f = open('column_3C_weka.csv',"r")
header = f.readline()
names = header.strip().split(',')[1:]
#print(names)
data1 = np.loadtxt(f ,delimiter=',', usecols=1+np.arange(5))

targetColumn = names.index("class")
XColumns = np.arange(4)
#XColumns = np.delete(XColumns, targetColumn)
X2 = data1[XColumns]
T2 = np.array(data1[5:7]).reshape((-1, 1)) # to keep 2-d matrix form
names.remove("class")

#print(X2)
#print(T2)

# Reading in file via pandas, and putting values into T(target) and X(inputs)
data = pandas.read_csv('column_3C_weka.csv')
T = data[['class']]
T = np.array(T)
X = data.drop(['class'], axis=1)
X = np.array(X)

#print(X)
#print(T)

# Getting labels for T and X
names = data.keys()
Xnames = names[3:27]
Tnames = names[0:2]
Xnames = Xnames.insert(0, 'bias')
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.8, classification=True, shuffle=True)
#Xtrain, Ttrain, Xtest, Ttest = ml.partition(X2, T2, 0.8, classification=True, shuffle=False)

print(Xtrain)
print(Ttrain)

import neuralnetworksA4 as nn
nnet = nn.NeuralNetworkClassifier(Xtrain.shape[0], [5,5], Ttrain.shape[1])
#print(nnet)
nnet.train(Xtrain.T, Ttrain, 200)


# In[1]:


import numpy as np
import mlutilities as ml
X = np.vstack((np.arange(20), [7, 4, 5, 5, 8, 4, 6, 7, 4, 9, 4, 2, 6, 6, 3, 3, 7, 2, 6, 4])).T
T = np.array([1]*8 + [2]*8 + [3]*4).reshape((-1, 1))
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.8, classification=True, shuffle=False)


# In[ ]:


T


# In[4]:


X


# In[5]:


Xtrain


# In[4]:


nnet.getErrors()[-1]


# In[3]:


import neuralnetworksA4 as nn
nnet = nn.NeuralNetworkClassifier(2, [5, 5], 3)
nnet.train(Xtrain, Ttrain, 200)
Ytest = nnet.use(Xtest)
fractionCorrect = np.sum(Ytest == Ttest) / len(Ttest)
print((Ytest.shape,Ttest.shape))
print(fractionCorrect)


# In[5]:


Ytest = nnet.use(Xtest)
fractionCorrect = (np.sum(Ytest == Ttest) / len(Ttest))
print((Ytest,Ttest))
print(fractionCorrect)


# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs445/notebooks/A4grader.tar) and extract A4grader.py from it.   

# In[ ]:


get_ipython().magic('run -i A4grader.py')


# # Extra Credit

# Earn 1 extra credit point by doing a few experiments with different neural network classifiers using the ReLU activation function on the orthopedic data. Discuss any differences you see from your earlier results that used tanh.
