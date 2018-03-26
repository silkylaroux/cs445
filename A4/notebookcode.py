
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


# In[6]:


X = np.arange(10).reshape((-1, 1))
T = np.array([1]*5 + [2]*5).reshape((-1, 1))


# In[16]:


netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(X, T, 20)
print(netc)
print('T, Predicted')
print(T)
print(netc.use(X))
print(np.hstack((T, netc.use(X).reshape(-1,1))))


# ## Addition to ```partition``` function

# Add the keyword parameter ```classification``` with a default value of ```False``` to your ```partition``` function.  When its value is set to ```True``` your ```partition``` function must perform a stratified partitioning as illustrated in lecture notes [12 Introduction to Classification](http://nbviewer.ipython.org/url/www.cs.colostate.edu/~anderson/cs445/notebooks/12%20Introduction%20to%20Classification.ipynb)     

# In[17]:


import mlutilities as ml
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.6, classification=True)


# In[18]:


Xtrain


# In[19]:


Ttrain


# In[20]:


Xtest


# In[21]:


Ttest


# ## Printing a confusion matrix

# Add these two functions to your ```mlutilities.py``` file.

# In[22]:


def confusionMatrix(actual, predicted, classes):
    nc = len(classes)
    confmat = np.zeros((nc, nc)) 
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        keep = trues
        predictedThisClassAboveThreshold = predictedThisClass
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
    printConfusionMatrix(confmat,classes)
    return confmat

def printConfusionMatrix(confmat,classes):
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('{:s}'.format('------'*len(classes)))
    for i,t in enumerate(classes):
        print('{:2d} |'.format(t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('{:5.1f}'.format(100*confmat[i,i1]), end='')
        print()


# ### Example with toy data

# Use the above data to compare QDA, LDA, and linear and nonlinear logistic regression.

# In[23]:


import qdalda
qda = qdalda.QDA()
qda.train(Xtrain, Ttrain)
Ytrain = qda.use(Xtrain)
Ytest = qda.use(Xtest)


# In[24]:


print(np.hstack((Ttrain, Ytrain)))


# In[26]:


np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[27]:


print(np.hstack((Ttest, Ytest)))


# In[28]:


np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[29]:


lda = qdalda.LDA()
lda.train(Xtrain, Ttrain)
Ytrain = lda.use(Xtrain)
Ytest = lda.use(Xtest)


# In[30]:


print(np.hstack((Ttrain, Ytrain)))


# In[14]:


print(np.hstack((Ttest, Ytest)))


# In[31]:


np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[32]:


np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[33]:


ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[34]:


ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# In[35]:


netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[37]:


print(np.hstack((Ttrain, Ytrain)))


# In[38]:


print(np.hstack((Ttest, Ytest)))


# In[39]:


np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[40]:


np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[41]:


ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[11]:


ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# Remember that linear logistic regression can be applied by specifying 0 hidden units.

# In[42]:


netc = nn.NeuralNetworkClassifier(X.shape[1], 0, len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[27]:


ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[43]:


ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# ## Apply to data from orthopedic patients

# Download ```column_3C_weka.csv``` from [this Kaggle site](https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients).  Use the column named ```class``` to create your target class labels. Apply QDA, LDA, linear logistic regression, and nonlinear logistic regression to this data.  Experiment with different hidden layer structures and numbers of iterations and describe what you find.
# 
# Partition data into 80% for training and 20% for testing, with ```shuffle=True```.
# 
# Print percents of training and testing samples correctly classified by QDA, LDA and various neural network classifiers.  Also print confusion matrices for training and for testing samples for each classifier.  Discuss the relative performance of your classifiers.

# In[50]:


import numpy as np
import mlutilities as ml
X = np.vstack((np.arange(20), [7, 4, 5, 5, 8, 4, 6, 7, 4, 9, 4, 2, 6, 6, 3, 3, 7, 2, 6, 4])).T
T = np.array([1]*8 + [2]*8 + [3]*4).reshape((-1, 1))
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.8, classification=True, shuffle=False)


# In[51]:


X


# In[52]:


Xtrain


# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs445/notebooks/A4grader.tar) and extract A4grader.py from it.   

# In[53]:


get_ipython().magic('run -i A4grader.py')


# # Extra Credit

# Earn 1 extra credit point by doing a few experiments with different neural network classifiers using the ReLU activation function on the orthopedic data. Discuss any differences you see from your earlier results that used tanh.
