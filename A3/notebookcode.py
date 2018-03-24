
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

# # Assignment 3: Activation Functions

# Damian Armijo

# ## Overview

# In this assignment, you will make a new version of your ```NeuralNetwork``` class from the previous assignment. For this new version, define the activation function to be the Rectified Linear Unit (ReLU).
# 
# You will compare the training and testing performances of networks with tanh and networks with the ReLU activation functions.

# ### NeuralNetworkReLU

# Start with the ```NeuralNetwork``` class defined in ```neuralnetworksA2.py```.  Define a new class named ```NeuralNetworkReLU``` that extends ```NeuralNetwork``` and simply defines new implementations of ```activation``` and ```activationDerivative``` that implement the ReLU activation function.

# ### Comparison

# Define a new function ```partition``` that is used as this example shows.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

X = np.arange(10*2).reshape((10, 2))
T = X[:, 0:1] * 0.1


# In[2]:


import random
def partition(X, T, seeding,shuffle = False):
    nRows = X.shape[0]
    rows = np.arange(nRows)
    if(shuffle):
        np.random.shuffle(rows)

    nTrain = int(nRows * seeding)
    trainRows = rows[:nTrain]
    testRows = rows[nTrain:]
    Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
    Xtest, Ttest = X[testRows, :], T[testRows, :]
    return Xtrain, Ttrain, Xtest, Ttest


# In[ ]:


X


# In[ ]:


T


# In[ ]:


Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8, shuffle=False)


# In[ ]:


Xtrain


# In[ ]:


Ttrain


# In[ ]:


Xtest


# In[ ]:


Ttest


# If ```shuffle=True``` is used as an argument, then the samples are randomly rearranged before the partitions are formed.

# In[ ]:


Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8, shuffle=True)


# In[ ]:


Xtrain


# In[ ]:


Ttrain


# In[ ]:


Xtest


# In[ ]:


Ttest


# You will use the ```energydata_complete.csv``` data for the following comparisons.  Load this data using pandas, then create matrix $X$ using all columns except ```['date','Appliances', 'rv1', 'rv2']``` and create $T$ using just ```'Appliances'```.  Write python code that performs the following algorithm.

# In[3]:


import pandas

# Reading in file via pandas, and putting values into T(target) and X(inputs)
data = pandas.read_csv('energydata_complete.csv')
T = data[['Appliances']]
T = np.array(T)
X = data.drop(['date','Appliances', 'rv1', 'rv2'], axis=1)
X = np.array(X)



# Getting labels for T and X
names = data.keys()
Xnames = names[3:27]
Tnames = names[0:2]
Xnames = Xnames.insert(0, 'bias')

print(X)
print(T)


# In[4]:


def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))


#   - For each of the two activation functions, ```tanh```, and ```ReLU```:
#       - For each hidden layer structure in [[u]*nl for u in [1, 2, 5, 10, 50] for nl in [1, 2, 3, 4, 5, 10]]:
#           - Repeat 10 times:
#               - Randomly partition the data into training set with 80% of samples and testing set with other 20%.
#               - Create a neural network using the given activation function and hidden layer structure.
#               - Train the network for 100 iterations.
#               - Calculate two RMS errors, one on the training partition and one on the testing partitions.
#           - Calculate the mean of the training and testing RMS errors over the 10 repetitions.

# In[5]:


import neuralnetworksA2 as nn


# In[9]:


def meanFromData(V):
    tanhMeanTrain = V[:,0].mean()
    tanhMeanTest = V[:,1].mean()
    reluMeanTrain = V[:,2].mean()
    reluMeanTest = V[:,3].mean()
    return tanhMeanTrain, tanhMeanTest,reluMeanTrain,reluMeanTest;


# In[10]:


#ReLU activation
import pandas as pd
errors = []
hiddens = [0] + [[nu] * nl for nu in [1,2,5,10] for nl in [1,2,3,4]]
V = np.zeros(shape=(2,4))
for hids in hiddens:
    for x in range(2):
        Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8, shuffle=True)
        nnet = nn.NeuralNetwork(Xtrain.shape[1], hids, Ttrain.shape[1])
        nnet.train(Xtrain, Ttrain, 100)
        stack = [rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))]
        
        nnetrelu = nn.NeuralNetworkReLU(Xtrain.shape[1], hids, Ttrain.shape[1])
        nnetrelu.train(Xtrain, Ttrain, 100)
        stack.extend([rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])
        
        V = np.vstack([V,stack])
    
    tanhMeanTrain,tanhMeanTest,reluMeanTrain,reluMeanTest = meanFromData(V)
    errors.append([hids, tanhMeanTrain,tanhMeanTest,reluMeanTrain,reluMeanTest])    
errors = pd.DataFrame(errors)
print(errors)

plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('tanh Train RMSE','tanh Test RMSE','ReLU Train RMSE', 'ReLU Test RMSE',))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.grid(True)


# You will have to add steps in this algorithm to collect the results you need to make the following plot.
# 
# Make a plot of the RMS errors versus the hidden layer structure.  On this plot include four curves, for the training and testing RMS errors for each of the two activation functions.  Label both axes and add a legend that identifies each curve.
# 
# As always, discuss what you see.  What can you say about which activation function is best?

# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A3grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A3grader.tar) and extract `A3grader.py` from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect execution score of  60 / 60 if your functions and class are defined correctly. The remaining 40 points will be based on the results you obtain from the comparisons of hidden layer structures and the two activation functions applied to the energy data.
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A3.ipynb` with `Lastname` being your last name, and then save this notebook.  Your working director must also contain `neuralnetworksA2.py` and `mlutilities.py` from lecture notes.
# 
# Combine your notebook, `neuralnetworkA2.py`, and `mlutilities.py` into one zip file or tar file.  Name your tar file `Lastname-A3.tar` or your zip file `Lastname-A3.zip`.  Check in your tar or zip file using the `Assignment 3` link in Canvas.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include other tests.

# In[2]:


get_ipython().magic('run -i A3grader.py')


# In[1]:


import neuralnetworksA2 as nn
nnet = nn.NeuralNetwork(1, 10, 1)
nnetrelu = nn.NeuralNetworkReLU(1, 5, 1)
da = nnetrelu.activation(-0.8)
print(da)

