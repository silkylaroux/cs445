import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '2'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
    # import A2mysolution as useThisCode
    # train = useThisCode.train
    # trainSGD = useThisCode.trainSGD
    # use = useThisCode.use
    # rmse = useThisCode.rmse
else:
    import subprocess, glob, pathlib
    filename = next(glob.iglob('*-A{}.ipynb'.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         '*-A{}.ipynb'.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.ClassDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

def close(a, b, within=0.01):
    return abs(a-b) < within

g = 0

print('\nTesting  import neuralnetworksA2 as nn')
points = 10
try:
    import neuralnetworksA2 as nn
    g += points
    print('\n--- {}/{} points. The statement  import neuralnetworksA2 as nn  works.'.format(points, points))
except Exception as ex:
    print('\n--- 0/{} points. The statement  import neuralnetworksA2 as nn  does not work.'.format(points))
    
print('''\nTesting nnet = nn.NeuralNetwork(1, 10, 1)''')
points = 10
try:
    nnet = nn.NeuralNetwork(1, 10, 1)
    g += points
    print('\n--- {}/{} points. nnet correctly constructed'.format(points, points))
except Exception as ex:
    print('\n--- 0/{} points. nnet not correctly constructed'.format(points))

print('''\nTesting a = nnet.activation(-0.8)''')
points = 10
try:
    a = nnet.activation(-0.8)
    correctAnswer = -0.664
    if close(a, correctAnswer):
        g += points
        print('\n--- {}/{} points. activation of {} is correct.'.format(points, points, a))
    else:
        print('\n--- 0/{} points. activation of {} is not correct. It should be {}.'.format(points, a, correctAnswer))

except Exception as ex:
    print('\n--- 0/{} points. The call to activation raised exception\n {}'.format(points, ex))
    

print('''\nTesting da = nnet.activationDerivative(-0.664)''')
points = 10
try:
    a = nnet.activationDerivative(-0.664)
    correctAnswer = 0.55906
    if close(a, correctAnswer):
        g += points
        print('\n--- {}/{} points. activationDerivative of {} is correct.'.format(points, points, a))
    else:
        print('\n--- 0/{} points. activationDerivative of {} is not correct. It should be {}.'.format(points, a, correctAnswer))
except Exception as ex:
    print('\n--- 0/{} points. The call to activationDerivative raised exception\n {}'.format(points, ex))

print('''\nTesting X = np.arange(300).reshape((-1, 3))
        T = X[:,0:2] + 0.1 * X[:,1:2] * X[:,2:3]
        import neuralnetworksA2 as nn
        nnet = nn.NeuralNetwork(3, [20, 10], 2)
        nnet.train(X, T, 100)
        error = np.sqrt(np.mean((T - nnet.use(X))**2))''')

X = np.arange(300).reshape((-1, 3))
T = X[:,0:2] + 0.1 * X[:,1:2] * X[:,2:3]

points = 20
try:
    nnet = nn.NeuralNetwork(3, [50, 10], 2)
    nnet.train(X, T, 1000)
    error = np.sqrt(np.mean((T - nnet.use(X))**2))
    if error < 2.0:
        g += points
        print('\n--- {}/{} points. The error of {} is correct.'.format(points, points, error))
    else:
        print('\n--- 0/{} points. The error of {} is incorrect. It should be less than 1.5.'.format(points, error))

except Exception as ex:
    print('\n--- 0/{} points. This exception was raised:\n {}'.format(points, ex))



name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {}/60'.format(name, g))

print('\n============= Training Result: Plots and Descriptions =============')

print('\n--- _/10 points. Comparison of hidden layer structures: plots.\nComments:')

print('''\n--- _/10 points. Comparison of hidden layer structures: 
         discussions of which structures appear to be the best and 
         how much it varies over multiple runs.\nComments:''')


print('\n--- _/10 points. Comparison of numbers of training iterations: plots.\nComments:')

print('''\n--- _/10 points. Comparison of numbers of training iterations:
         discussions of which number of iterations appear to be the best and 
         how much it varies over multiple runs.\nComments:''')



print('\n{} Notebook Grade is __ / 40'.format(name))

print('\n{} FINAL GRADE is __ / 100'.format(name))



