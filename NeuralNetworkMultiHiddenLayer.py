import numpy as np
import math
from scipy import optimize
import matplotlib.pyplot as plt
import time
import CSB_Data_Wrangle as wrangler

class WrongNetworkLayerSizesError(Exception):
    '''raise this when there is an error with the network size being imported'''
    pass

class Neural_Network:
    def __init__(self, input_layer_size, output_layer_size, hidden_layer_sizes=[], \
                 activation_func = 'sigmoid', import_weights=False, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = input_layer_size#len(inputs[0])
        self.outputLayerSize = output_layer_size#len(output[0])
        self.hiddenLayerSizes = hidden_layer_sizes
        self.layerSizes = [self.inputLayerSize] + self.hiddenLayerSizes + [self.outputLayerSize]
        self.num_layers = len(self.hiddenLayerSizes) + 2
        self.weights = []
        self.bias_weights = []
        self.Lambda = Lambda

        if activation_func == 'ELU':
            self.activ_func = self.elu
            self.activ_func_prime = self.eluPrime
        elif activation_func == 'RELU':
            self.activ_func = self.relu
            self.activ_func_prime = self.reluPrime
        else:
            self.activ_func = self.sigmoid
            self.activ_func_prime = self.sigmoidPrime


        #Initialize weights or retrieve weights from a file
        if not import_weights:
            for i, ls in enumerate(self.layerSizes):
                if i == self.num_layers - 1: break
                self.weights.append(np.random.normal(0,ls**-0.5,(self.layerSizes[i+1],ls)))
                self.bias_weights.append(np.random.normal(0,1,(self.layerSizes[i+1],1)))
        else:
            try:
                weights_file = np.load('weights.npz')
                for i, j in enumerate(weights_file):
                    if i == 0:
                        self.weights = weights_file[j]
                    else:
                        self.bias_weights.append(weights_file[j])
                layer_sizes_import = np.load('layer_sizes.npy')
                if any(layer_sizes_import != self.layerSizes): raise WrongNetworkLayerSizesError
            except IOError as e:
                print("File not found error: {0}".format(e))
                exit()
            except WrongNetworkLayerSizesError:
                print("The network being imported has a different size than \n the network you created.")
                exit()
        
    def forward(self, X, train_items=-1):
        #Propogate inputs though network
        self.z = []
        self.a = []
        if train_items == -1:
            train_sets = X.shape[0]
        else:
            train_sets = 1

        for i, w in enumerate(self.weights):
            bias_vector = np.ones((train_sets,1))
            bias = np.dot(bias_vector, self.bias_weights[i])
            if i == 0:
                self.z.append(np.dot(X, w) + bias)
                self.a.append(self.activ_func(self.z[-1]))
            else:
                self.z.append(np.dot(self.a[-1], w) + bias)
                self.a.append(self.activ_func(self.z[-1]))
        
        yHat = self.a[-1]
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def elu(self, z):
        alpha = 1.5
        return np.where(z <= 0, alpha*(np.exp(z) - 1), z)
    
    def eluPrime(self, z):
        alpha = 1.5
        return np.where(z <= 0, self.elu(z) + alpha, 1)

    def relu(self, z):
        return np.where(z < 0, 0, z)

    def reluPrime(self, z):
        return np.where(z < 0, 0, 1)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        train_sets = X.shape[0] if self.Lambda != 0 else 1
        J = (1/2)*np.sum((y-self.yHat)**2)/train_sets + (self.Lambda/2) * np.sum([np.sum(x**2) for x in self.weights])
        return J
        
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta = []
        dJdW = []
        dJdW_bias = []
        train_sets = X.shape[0] if self.Lambda != 0 else 1
        
        for i in range(self.num_layers - 2,-1,-1):
            if i == self.num_layers - 2:
                delta.insert(0, np.multiply(-(y-self.yHat), self.activ_func_prime(self.z[i])))
            else:
                delta.insert(0, np.dot(delta[0], self.weights[i+1].T) * self.activ_func_prime(self.z[i]))

            if i == 0:
                dJdW.insert(0, np.dot(X.T, delta[0])/train_sets + self.Lambda * self.weights[i])
            else:
                dJdW.insert(0, np.dot(self.a[i-1].T, delta[0])/train_sets + self.Lambda * self.weights[i])
            
            bias_vector = np.ones((1,X.shape[0]))
            dJdW_bias.insert(0, np.dot(bias_vector, delta[0])/train_sets + self.Lambda * self.bias_weights[i]) #Bias dJdW updated as just the delta

        return (dJdW, dJdW_bias)
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        temp = []
        for x, y in zip(self.weights, self.bias_weights):
            temp.append(np.concatenate((x.ravel(), y.ravel())))
        params = np.concatenate(temp)
        return params
    
    def setParams(self, params):
        
        W_start, W_end = 0, 0
        W_start_bias, W_end_bias = 0, 0
        for i in range(self.num_layers - 1):
            front_end_layer, back_end_layer = i, i + 1

            W_end = W_start + self.layerSizes[front_end_layer] * self.layerSizes[back_end_layer]
            W_start_bias = W_end
            W_end_bias = W_start_bias + self.layerSizes[back_end_layer]
            try:
                self.weights[i] = np.reshape(params[W_start:W_end], \
                    (self.layerSizes[front_end_layer], self.layerSizes[back_end_layer]))
            except:
                pass
            self.bias_weights[i] = np.reshape(params[W_start_bias:W_end_bias], \
                (1, self.layerSizes[back_end_layer]))
            W_start = W_end_bias

    def computeGradients(self, X, y):
        dJdW, dJdW_bias = self.costFunctionPrime(X, y)
        temp = []
        for x, y in zip(dJdW, dJdW_bias):
            temp.append(np.concatenate((x.ravel(), y.ravel())))
        grad = np.concatenate(temp)
        return grad

class manual_trainer:
    def __init__(self, N):
        self.N = N

    def train(self, trainX, trainY, testX, testY, iteration_limit = 200, time_limit = 0, learning_rate = 1):
        self.X = trainX
        self.y = trainY
        self.testX = testX
        self.testY = testY
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.J = []
        self.J_test = []
        self.learning_rate = learning_rate

        training_limit_reached = False
        iterations = 0
        start_time = time.time()

        while not training_limit_reached:
            iterations += 1

            dJdW, dJdW_bias = self.N.costFunctionPrime(self.X, self.y)
            for i in range(self.N.num_layers-1):
                self.N.weights[i] -= dJdW[i] * (1-iterations/iteration_limit) * self.learning_rate
                self.N.bias_weights[i] -= dJdW_bias[i] * (1-iterations/iteration_limit) * self.learning_rate * 1/2

            self.J.append(self.N.costFunction(self.X, self.y))
            self.J_test.append(self.N.costFunction(self.testX, self.testY))

            if time_limit == 0 and iterations >= iteration_limit: training_limit_reached = True
            if time_limit != 0 and (time.time() - start_time) > time_limit: training_limit_reached = True


class trainer:
    def __init__(self, N):
        self.N = N
        self.count_train = 0

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.J_test.append(self.N.costFunction(self.testX, self.testY))
        self.count_train += 1
        if self.count_train % 25 == 0: print(self.count_train)

    def train(self, trainX, trainY, testX, testY, max_iter=200, learning_rate = 1):
        self.X = trainX
        self.testX = testX
        self.y = trainY
        self.testY = testY
        self.J = []
        self.J_test = []
        self.learning_rate = learning_rate

        params0 = self.N.getParams()

        options = {'maxiter': max_iter, 'disp': True}

        _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True,
                                 method='BFGS', args = (trainX,trainY), options=options,
                                  callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

# inputs = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
# output = np.array(([75], [82], [93], [70]), dtype=float)

# inputs = inputs/np.amax(inputs, axis=0)
# output = output/100

# inputs_test = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
# output_test = np.array(([70], [89], [85], [75]), dtype=float)

# inputs_test = inputs_test/np.amax(inputs_test, axis=0)
# output_test = output_test/100


start_time = time.time()
# inputs_big = np.array([[(x * math.pi) / 33] for x in range(0,34)])
# output_big = np.sin(inputs_big)
# inputs_big = inputs_big / math.pi

# inputs = np.array([[(x * math.pi) / 16] for x in range(0,17)])
# output = np.sin(inputs)

# inputs = inputs / math.pi

# inputs_test = np.array([[(x * math.pi) / 16] for x in range(1,17,2)])
# output_test = np.sin(inputs_test)

# inputs_test = inputs_test / math.pi

folder_name = '/TrainingData-ExportFromCSB/'
data = wrangler.GetData(folder_name)

input_norm = data[0]
output_norm = data[1]
input_min = data[2]
input_max = data[3]

split_point = int(len(input_norm) * 0.8)
input_split = np.split(input_norm,[split_point])
output_split = np.split(output_norm,[split_point])

inputs = input_split[0]
inputs_test = input_split[1]
output = output_split[0]
output_test = output_split[1]

input_layer_nodes = 20
hidden_layer_nodes = [8,8,8]
output_layer_nodes = 2
learning_rate = 1
Lambda = 6e-6#0.0000001 aka 1e-7
import_weights = False
save_weights = True
train = True

NN = Neural_Network(input_layer_nodes, output_layer_nodes, hidden_layer_nodes, \
                    activation_func='sigmoid', Lambda = Lambda, import_weights = import_weights)
numgrad = computeNumericalGradient(NN, inputs, output)
grad = NN.computeGradients(inputs, output)
print(np.linalg.linalg.norm(grad-numgrad)/np.linalg.linalg.norm(grad+numgrad))
if train:
    T = trainer(NN)
    T.train(inputs,output,inputs_test,output_test,max_iter=1500,learning_rate = learning_rate)
    # T = manual_trainer(NN)
    # T.train(inputs,output,inputs_test,output_test,iteration_limit=500,learning_rate=1)
    print((time.time() - start_time))
    plt.plot(T.J)
    plt.plot(T.J_test)
    plt.grid(1)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.ylim(0,0.5)
    plt.show()

NN_calc_output_test = np.clip(NN.forward(inputs_test[0:40]), 0, 1)
NN_target_output_test = output_test[0:40]

NN_calc_output = np.clip(NN.forward(inputs[0:40]), 0, 1)
NN_target_output = output[0:40]

# print(NN_calc_output)
# print(NN_target_output)

# print(NN.costFunctionPrime(inputs,output))
# print((NN.forward(inputs_big) - output_big)*100)
# print(NN.forward(inputs))

diff = NN_calc_output_test[:,0] - NN_target_output_test[:,0]
diff2 = NN_calc_output[:,0] - NN_target_output[:,0]
diff *= 72
diff2 *= 72

print("hdg change std dev: ",round(np.std(diff)/2))

plt.plot(diff, 'ro')
plt.plot(diff2, 'b.')
plt.axis([0, len(NN_calc_output_test), -36, 36])
plt.grid(True)
# plt.plot(inputs,NN.forward(inputs))
# plt.plot(inputs_big,output_big)
# plt.plot(inputs_test,NN.forward(inputs_test))
plt.show()

diff = NN_calc_output_test[:,1] - NN_target_output_test[:,1]
diff2 = NN_calc_output[:,1] - NN_target_output[:,1]
diff *= 400
diff2 *= 400

print("thrust std dev: ",round(np.std(diff)/2))

plt.plot(diff, 'ro')
plt.plot(diff2, 'b.')
plt.axis([0, len(NN_calc_output_test), -200, 200])
plt.grid(True)
plt.show()

if save_weights:
    np.set_printoptions(linewidth=1000000)
    
    with open('input_norm_min_max.txt', 'w') as f:
        f.write(np.array_repr(np.array([input_min,input_max]), \
        max_line_width=1000000).replace("array","np.array").replace(", dtype=object","") \
        +'\n')
    
    with open('weights.txt', 'w') as f:
        f.write(np.array_repr(np.array(NN.weights), \
            max_line_width=1000000).replace("array","np.array").replace(", dtype=object","") \
            +'\n')

    with open('bias_weights.txt', 'w') as f:
        s = "[\n"
        for i, w in enumerate(NN.bias_weights):
            s += np.array_repr(w, max_line_width=1000000).replace('array','np.array')
            if i != len(NN.bias_weights) - 1:
                s += ',\n'
            else:
                s += '\n]'
        f.write(s)
        


# if save_weights:
#     np.savez('weights',NN.weights,*NN.bias_weights)
#     np.save('layer_sizes',NN.layerSizes)
#     for i, w in enumerate(NN.weights):
#         np.savetxt('weights'+str(i)+'.txt',np.array_str(w),delimiter=',')
#         np.savetxt('bias_weights'+str(i)+'.txt',np.array_str(NN.bias_weights[i]),delimiter=',')



# w = np.load('weights.npz')

pass
# #Test network for various combinations of sleep/study:
# hoursSleep = np.linspace(0, 10, 100)
# hoursStudy = np.linspace(0, 5, 100)

# #Normalize data (same way training data way normalized)
# hoursSleepNorm = hoursSleep/10.
# hoursStudyNorm = hoursStudy/5.

# #Create 2-d versions of input for plotting
# a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

# #Join into a single input matrix:
# allInputs = np.zeros((a.size, 2))
# allInputs[:, 0] = a.ravel()
# allInputs[:, 1] = b.ravel()

# allOutputs = NN.forward(allInputs)

# #Contour Plot:
# yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
# xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

# CS = plt.contour(xx,yy,100*allOutputs.reshape(100, 100))
# plt.clabel(CS, inline=1, fontsize=10)
# plt.xlabel('Hours Sleep')
# plt.ylabel('Hours Study')
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.scatter(10*inputs[:,0], 5*inputs[:,1], 100*output, c='k', alpha = 1, s=30)
# ax.scatter(10*inputs_test[:,0], 5*inputs_test[:,1], 100*output_test, c='r', alpha = 1, s=30)

# surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100),cmap='jet')

# ax.set_xlabel('Hours Sleep')
# ax.set_ylabel('Hours Study')
# ax.set_zlabel('Test Score')
# plt.show()