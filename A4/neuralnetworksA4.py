import numpy as np
import sys
import mlutilities as ml
import matplotlib.pyplot as plt
from copy import copy
import time


class NeuralNetwork:

    def __init__(self, ni, nhs, no):
        if not isinstance(nhs, list) and not isinstance(nhs, tuple):
            nhs = [nhs]
        if nhs[0] == 0:  # assume nhs[0]==0 and len(nhs)>1 never True
            nihs = [ni]
            nhs = []
        else:
            nihs = [ni] + nhs

        if len(nihs) > 1:
            self.Vs = [1/np.sqrt(nihs[i]) *
                       np.random.uniform(-1, 1, size=(1+nihs[i], nihs[i+1])) for i in range(len(nihs)-1)]
            self.W = 1/np.sqrt(nihs[-1]) * np.random.uniform(-1, 1, size=(1+nihs[-1], no))
        else:
            self.Vs = []
            self.W = 1/np.sqrt(ni) * np.random.uniform(-1, 1, size=(1+ni, no))
        self.ni, self.nhs, self.no = ni, nhs, no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.trained = False
        self.reason = None
        self.errorTrace = None
        self.numberOfIterations = None
        self.trainingTime = None

    def __repr__(self):
        str = 'NeuralNetwork({}, {}, {})'.format(self.ni, self.nhs, self.no)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations that took {:.4f} seconds. Final error is {}.'.format(self.numberOfIterations, self.getTrainingTime(), self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _pack(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self, w):
        first = 0
        numInThisLayer = self.ni
        for i in range(len(self.Vs)):
            self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1, self.nhs[i]))
            first += (numInThisLayer+1) * self.nhs[i]
            numInThisLayer = self.nhs[i]
        self.W[:] = w[first:].reshape((numInThisLayer+1, self.no))

    def _objectiveF(self, w, X, T):
        self._unpack(w)
        # Do forward pass through all layers
        Zprev = X
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :])  # handling bias weight without adding column of 1's
                                                                   # Changed so that this calls activation not np.tanh()
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        return 0.5 * np.mean((T-Y)**2)

    def _gradientF(self, w, X, T):
        self._unpack(w)
        # Do forward pass through all layers
        Zprev = X
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :]) # Changed so that this calls activation not np.tanh()
            Z.append(Zprev)
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        # Do backward pass, starting with delta in output layer
        delta = -(T - Y) / (X.shape[0] * T.shape[1])
        dW = np.vstack((np.ones((1, delta.shape[0])) @ delta, 
                        Z[-1].T @ delta))
        dVs = []
        delta = (self.activationDerivative(Z[-1])) * (delta @ self.W[1:, :].T) # Changed so that this calls activationDerivative()
        for Zi in range(len(self.nhs), 0, -1):
            Vi = Zi - 1  # because X is first element of Z
            dV = np.vstack((np.ones((1, delta.shape[0])) @ delta,
                            Z[Zi-1].T @ delta))
            dVs.insert(0, dV)
            delta = (delta @ self.Vs[Vi][1:, :].T) * (self.activationDerivative(Z[Zi-1])) # Changed so that this calls                                                                                                               # activationDerivative()
        return self._pack(dVs, dW)

    def train(self, X, T, nIterations=100, verbose=False,
              weightPrecision=0, errorPrecision=0, saveWeightsHistory=False):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self._standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self._standardizeT(T)

        startTime = time.time()

        scgresult = ml.scg(self._pack(self.Vs, self.W),
                            self._objectiveF, self._gradientF,
                            X, T,
                            xPrecision=weightPrecision,
                            fPrecision=errorPrecision,
                            nIterations=nIterations,
                            verbose=verbose,
                            ftracep=True,
                            xtracep=saveWeightsHistory)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        #self.errorTrace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to _unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        self.weightsHistory = scgresult['xtrace'] if saveWeightsHistory else None
        self.trainingTime = time.time() - startTime
        return self

    def use(self, X, allOutputs=False):
        Zprev = self._standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :]) # Changed so that this calls activation not np.tanh()
            Z.append(Zprev)
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        Y = self._unstandardizeT(Y)
        return (Y, Z[1:]) if allOutputs else Y
    
    def activation(self, weighted_sum):
        return np.tanh(weighted_sum)
    
    def activationDerivative(self, activation_value):
        return 1 - activation_value * activation_value
    
    def getNumberOfIterations(self):
        return self.numberOfIterations

    def getErrors(self):
        return self.errorTrace

    def getTrainingTime(self):
        return self.trainingTime

    def getWeightsHistory(self):
        return self.weightsHistory

    def draw(self, inputNames=None, outputNames=None, gray=False):
        ml.draw(self.Vs, self.W, inputNames, outputNames, gray)

    def _multinomialize(self, Y):   # also known as softmax
        # fix to avoid overflow
        mx = max(0, np.max(Y))
        expY = np.exp(Y - mx)
        denom = np.sum(expY, axis=1).reshape((-1, 1)) + sys.float_info.epsilon
        return expY / denom
        
class NeuralNetworkReLU(NeuralNetwork):
    
    def activation(self, weighted_sum):
        return np.maximum(weighted_sum, 0)
    
    def activationDerivative(self, weighted_sum):
        return 1. * (weighted_sum > 0)

class NeuralNetworkClassifier(NeuralNetwork):

    def _multinomialize(self, Y):   # also known as softmax
        # fix to avoid overflow
        mx = max(0, np.max(Y))
        expY = np.exp(Y - mx)
        denom = np.sum(expY, axis=1).reshape((-1, 1)) + sys.float_info.epsilon
        return expY / denom
    
    def loglikelihood(warg,K):
        w = warg.reshape((-1,K))
        Y = g(Xtrains1,w)
        # print(w)
        return - np.mean(TtrainI*np.log(Y))
    
    def g(X,w):
        fs = np.exp(np.dot(X, w))  # N x K
        denom = np.sum(fs,axis=1).reshape((-1,1))
        # pdb.set_trace()
        gs = fs / denom
        # print(gs[:10,:])
        return gs
    
    def _objectiveF(self, w, X, Tindicators):
        self._unpack(w)
        # Do forward pass through all layers
        Zprev = X
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :])  # handling bias weight without adding column of 1's
                                                                   # Changed so that this calls activation not np.tanh()
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        #Y = self.g(X[0:],w);
        #return 0.5 * np.mean((T-Y)**2)
        
        G = self._multinomialize(Y)
        #print(("tind",Tindicators,"g",G,"ep",np.log(G + sys.float_info.epsilon)))
        return -np.mean(Tindicators * np.log(G + sys.float_info.epsilon))

        #return np.log((G + sys.float_info.epsilon))
        # Convert Y to multinomial distribution. Call result G.
        #  and return negative of mean log likelihood.
        # Write the call to np.log as  np.log(G + sys.float_info.epsilon)
    def catchNegative(x):
        return x if x > 0 else sys.float_info.epsilon
    
    def _gradientF(self, w, X, Tindicators):
        self._unpack(w)
        # Do forward pass through all layers
        Zprev = X
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :]) # Changed so that this calls activation not np.tanh()
            Z.append(Zprev)
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        #print(Y)
        
        G = self._multinomialize(Y)
        
        # Do backward pass, starting with delta in output layer
        #print(Tindicators, G)
        delta = -(Tindicators - G) / (X.shape[0] * G.shape[1])
        #print(("delt",delta))
        dW = np.vstack((np.ones((1, delta.shape[0])) @ delta, 
                        Z[-1].T @ delta))
        dVs = []
        delta = (self.activationDerivative(Z[-1])) * (delta @ self.W[1:, :].T) # Changed so that this calls activationDerivative()
        for Zi in range(len(self.nhs), 0, -1):
            Vi = Zi - 1  # because X is first element of Z
            dV = np.vstack((np.ones((1, delta.shape[0])) @ delta,
                            Z[Zi-1].T @ delta))
            dVs.insert(0, dV)
            delta = (delta @ self.Vs[Vi][1:, :].T) * (self.activationDerivative(Z[Zi-1])) # Changed so that this calls                                                                                                               # activationDerivative()
        #print(dVs,dW)
        return self._pack(dVs, dW)
        
    
        # Convert Y to multinomial distribution. Call result G.
        # delta calculated as before, except use Tindicators instead of T to
        #   make error of  Tindicates - G
        # Also, negate the result.  Why?

    


    def use(self, X, allOutputs=False):
        
        Zprev = self._standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = self.activation(Zprev @ V[1:, :] + V[0:1, :]) # Changed so that this calls activation not np.tanh()
            Z.append(Zprev)
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        #Y = self._unstandardizeT(Y)
        #return (Y, Z[1:]) if allOutputs else Y
        
        G = self._multinomialize(Y)
        #print(Y)
        #print(np.argmax(Y))
        #print(self.classes[G])
        #print(("y",Y,"g",G))
        # multinomialize Y, assign to G
        # Calculate predicted classes by
        #   picking argmax for each row of G
        #   and use the results to index into self.classes.
        #print([np.argmax(G,axis = 1)])
        tempclass = np.argmax( G, axis=1 ).reshape(-1,1)
        
        #print(("tc",tempclass))
        #print(("classes",self.classes,"other",self.classes[np.array(tempclass)]))
        self.classes= tempclass
        return (self.classes, G, Z[1:]) if allOutputs else self.classes
    
    def train(self, X, T, nIterations=100, verbose=False,
              weightPrecision=0, errorPrecision=0, saveWeightsHistory=False):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self._standardizeX(X)
        T = T.reshape((-1, 1))
        #if T.ndim == 1:
            #T = T.reshape((-1, 1))

        #if self.Tmeans is None:
            #self.Tmeans = T.mean(axis=0)
            #self.Tstds = T.std(axis=0)
            #self.Tconstant = self.Tstds == 0
            #self.TstdsFixed = copy(self.Tstds)
            #self.TstdsFixed[self.Tconstant] = 1

        startTime = time.time()
        Tindicators = ml.makeIndicatorVars(T)
        #print(Tindicators)
        self.classes = np.unique(T).reshape(-1,1)
        scgresult = ml.scg(self._pack(self.Vs, self.W),
                            self._objectiveF, self._gradientF,
                            X, Tindicators,
                            xPrecision=weightPrecision,
                            fPrecision=errorPrecision,
                            nIterations=nIterations,
                            verbose=verbose,
                            ftracep=True,
                            xtracep=saveWeightsHistory)
        
        
        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace'] 
        # * self.Tstds # to _unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        self.weightsHistory = scgresult['xtrace'] if saveWeightsHistory else None
        self.trainingTime = time.time() - startTime
        
        # Remove T standardization calculations
        # Assign to Tindicators using ml.makeIndicatorVars
        # Add   self.classes = np.unique(T)
        # Pass Tindicators into ml.scg instead of T
        return self

    
        
if __name__ == '__main__':

    X = np.arange(10).reshape((-1, 1))
    T = X + 2

    net = NeuralNetwork(1, 0, 1)
    net.train(X, T, 10)
    print(net)
    
    net = NeuralNetwork(1, [5, 5], 1)
    net.train(X, T, 10)
    print(net)