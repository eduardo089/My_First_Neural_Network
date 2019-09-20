import numpy as np

def diff_cost_function_L(y,a):
    return y-a
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def diff_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

#def update_weights(eta, weights, ):
#    weights = weights - nabla*

X = np.array([  [0,1],
                [0,1],
                [1,0],
                [1,0] ])
             
y = np.array([[0,0,1,1]]).T
sizes = [2,4,1]

#Inicializamos las variables
b = [np.random.randn(i,1) for i in sizes[:]]
w = [np.random.randn(k,j) for j,k in zip(sizes[:-1],sizes[1:])]
eta = 0.2
z = [np.zeros(i) for i in sizes [:]]
a = [np.zeros(i) for i in sizes [:]]
delta = [np.zeros(i) for i in sizes [:]]

data = 0
cost = 10
while data < len(y):
    while cost > 0.001:
        #Fill initial data
        for j in range(len(z[0])):
            z[0][j] = X[data][j]
            b[0][j] = X[data][j]
            a[0][j] = sigmoid(X[data][j])
        #Forward propagation
        for l in range(1,len(z)):
            for j in range(len(z[l])):
                #Recordar que sigmoid(z[l-1][:]) es lo mismo que a[l-1][:]
                z[l][j] = np.dot(w[l-1][j,:],sigmoid(z[l-1][:]))+b[l][j]
                a[l][j] = sigmoid(z[l][j])
        #Calculamos el Error
        for j in range(len(delta[-1])):
            delta[-1][j] = diff_cost_function_L(y[data],a[-1][j])*diff_sigmoid(z[-1][j])
        cost = 0.5*np.square((y[data] - a[-1][0]))
        print("Cost is {}".format(cost))
        
        #Backward Propagation
        #es el producto punto entre los pesos que "salen de la red" y los deltas siguientes
        for l in reversed(range(len(delta)-1)):
            for j in range(len(delta[l])):
                delta[l][j] = np.dot(w[l][:,j],delta[l+1][:])*diff_sigmoid(z[l][j])
                
        ##Actualización de biases y pesos
        for l in range(len(w)):
            for j in range(len(w[l])):
                for k in range(len(w[l][j])):
                    w[l][j][k] = w[l][j][k] - eta*a[l][k]*delta[l+1][j]
        
        for l in range(len(b)):
            for j in range(len(b[l])):
                b[l][j] = b[l][j] - eta*delta[l][j]
    data += 1

#predicción
y_hat = np.zeros(len(y))
for i in range(len(y)):
    for j in range(len(z[0])):
        z[0][j] = X[i][j]
        b[0][j] = X[i][j]
        a[0][j] = sigmoid(X[i][j])
    #Forward propagation
    for l in range(1,len(z)):
        for j in range(len(z[l])):
            #Recordar que sigmoid(z[l-1][:]) es lo mismo que a[l-1][:]
            z[l][j] = np.dot(w[l-1][j,:],sigmoid(z[l-1][:]))+b[l][j]
            a[l][j] = sigmoid(z[l][j])
    #Calculamos el Error
    for j in range(len(delta[-1])):
        delta[-1][j] = diff_cost_function_L(y[i],a[-1][j])*diff_sigmoid(z[-1][j])
    y_hat[i] = a[-1][0]
    cost = 0.5*np.square((y[i] - a[-1][0]))
    print("Cost is {}".format(cost))

print("the prediction is {}".format(y_hat))
