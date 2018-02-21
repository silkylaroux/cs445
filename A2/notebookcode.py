
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

# # Assignment 2: Neural Network Regression

# Damian Armijo

# ## Overview

# The goal of this assignment is to learn about object-oriented programming in python and to gain some experience in comparing different sized neural networks when applied to a data set.
# 
# Starting with the ```NeuralNetwork``` class from the lecture notes, you will create one new version of that class, apply it to a data set, and discuss the results.

# ## Modified neuralnetworks class testing

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[8]:


import neuralnetworksA2 as nn2

nnet = nn2.NeuralNetwork(1, [10], 1)


# In[9]:


[nnet.activation(s) for s in [-2, -0.5, 0, 0.5, 2]]


# In[10]:


[nnet.activationDerivative(nnet.activation(s)) for s in [-2, -0.5, 0, 0.5, 2]]


# In[11]:


nnet.train(X, T, 100, verbose=True)
nnet


# In[30]:


plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(nnet.getErrors())

plt.subplot(3, 1, 2)
plt.plot(X, T, 'o-', label='Actual')
plt.plot(X, nnet.use(X), 'o-', label='Predicted')

plt.subplot(3, 1, 3)
nnet.draw()


# ## Neural Network Performance with Different Hidden Layer Structures and Numbers of Training Iterations

# ### Example with Toy Data

# Using your new ```NeuralNetwork``` class, you can compare the error obtained on a given data set by looping over various hidden layer structures.  Here is an example using the simple toy data from above.

# In[13]:


import random

nRows = X.shape[0]
rows = np.arange(nRows)
np.random.shuffle(rows)
nTrain = int(nRows * 0.8)
trainRows = rows[:nTrain]
testRows = rows[nTrain:]
Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
Xtest, Ttest = X[testRows, :], T[testRows, :]


# In[14]:


Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[15]:


def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))


# In[18]:


import pandas as pd

errors = []
hiddens = [0] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
print('hiddens =', hiddens)
for hids in hiddens: 
    nnet = nn.NeuralNetwork(Xtrain.shape[1], hids, Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, 500)
    errors.append([hids, rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])
errors = pd.DataFrame(errors)
print(errors)

plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.grid(True)


# For this data and shuffling [5,5,5] had the lowest error at 0.452351. 
# The hightest error being 0.871766 for [1,1]. 
# The range of rmse's for the test data was .419415. 
# The median error value was 0.6325965. The above graph is used to show the hidden layer in the randomized toy data. 

# In[77]:


import pandas as pd
errors = []
nIterationsList = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
hiddens = [20,20,20,20] #[0] + [nu * nl for nu in nIterationsList for nl in [1]]

for hids in nIterationsList: 
    nnet = nn2.NeuralNetwork(Xtrain.shape[1], hiddens, Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, hids)
    errors.append([hids, rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])

errors = pd.DataFrame(errors)



print(nIterationsList)
print(errors)
plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), nIterationsList) #, rotation=30, horizontalalignment='right')
plt.grid(True)


# ### Experiments wtih Automobile Data

# The following section has:
# 
#   * cylinders,
#   * displacement,
#   * weight,
#   * acceleration,
#   * year, and
#   * origin
#   
# as input variables, and
# 
#   * mpg
#   * horsepower
#   
# as output variables.

# In[20]:


# This is a function which goes through and parses the auto-mpg.data file, and creates an input and target set from the data file
def makeMPGData(filename='auto-mpg.data'):
    np.set_printoptions(suppress=True)

    def missingIsNan(s):
        return np.nan if s == b'?' else float(s)
    data = np.loadtxt(filename, usecols=range(8), converters={3: missingIsNan}) # This chunk of code flags the input with nan values
    goodRowsMask = np.isnan(data).sum(axis=1) == 0
    data = data[goodRowsMask,:]
    
    X = data[:,1:3]
    otherX = data[:,4:]                  # This chunk of code takes the specific input values to be trained from data[]
    X = np.hstack((X,otherX))
    
    T = data[:,0:1]
    otherT = data[:,3:4]                 # This chunk of code takes the specific Target values from data[]
    T = np.hstack((T, otherT))
    
    Xnames =  ['cylinders','displacement','weight','acceleration','year','origin']
    Tnames = ['mpg', 'horsepower']
    return X,T,Xnames,Tnames


# In[21]:


# This line calls the previous method and stores the values that will be trained
X,T,Xnames,Tname = makeMPGData()


# In[22]:


# This section tests that the data was read in correctly, and that it can be implemented correctly into a neuralnetwork.
print("Shape of Target data: ", T.shape)
print("Shape of Input data: ", X.shape)
import neuralnetworksA2 as nn2

nnet = nn2.NeuralNetwork(6, [10,10,10], 2)

nnet.train(X, T, 100, verbose=True)
print(nnet)
error = np.sqrt(np.mean((T - nnet.use(X))**2))


# As can be seen above the neuralnetwork run with the data from the auto-mpg.data file is fast and with an error of approximately .193 after 100 iterations, it is fairly accurate. 

# In[26]:


# This section of code creates a randomized training set from 80 percent of the data given, and holds 20 percent of the data
# for testing. 
import random

nRows = X.shape[0]
rows = np.arange(nRows)
np.random.shuffle(rows)
nTrain = int(nRows * 0.8)
trainRows = rows[:nTrain]
testRows = rows[nTrain:]
Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
Xtest, Ttest = X[testRows, :], T[testRows, :]


# In[24]:


import pandas as pd
errors = []
hiddens = [0] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
print('hiddens =', hiddens)
for hids in hiddens: 
    nnet = nn2.NeuralNetwork(Xtrain.shape[1], hids, Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, 500)
    errors.append([hids, rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])
errors = pd.DataFrame(errors)
print(errors)

plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.grid(True)


# In[96]:


import pandas as pd
errors = []
nIterationsList = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
hiddens = [5,5,5,5] 

for hids in nIterationsList: 
    nnet = nn2.NeuralNetwork(Xtrain.shape[1], hiddens, Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, hids)
    errors.append([hids, rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])

errors = pd.DataFrame(errors)
#  ...  insert code here using the code in the previous code block as a guide ...

print(nIterationsList)
print(errors)
plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), nIterationsList) #, rotation=30, horizontalalignment='right')
plt.grid(True)


# In[8]:


plt.figure(figsize=(100, 100))

plt.subplot(6,10,3)
nnet.draw()


# Include the code, output, and graphs like the above examples. Discuss the results.  Investigate and discuss how much the best hidden layer structure and number of training iterations vary when you repeat the runs.

# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect execution score of  60 / 60 if your functions and class are defined correctly. The remaining 40 points will be based on the results you obtain from the comparisons of hidden layer structures and numbers of training iterations on the automobile data.
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.  Your working director must also contain `neuralnetworksA2.py` and `mlutilities.py` from lecture notes.
# 
# Combine your notebook and `neuralnetworkA2.py` into one zip file or tar file.  Name your tar file `Lastname-A2.tar` or your zip file `Lastname-A2.zip`.  Check in your tar or zip file using the `Assignment 2` link in Canvas.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include other tests.

# In[ ]:


get_ipython().magic('run -i A2grader.py')


# ## Extra Credit

# Repeat the comparisons of hidden layer structures and numbers of training iterations on a second data set from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml).
