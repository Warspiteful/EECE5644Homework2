import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Training")
    xTrain = data[:, 0:2]
    yTrain = data[:, 2]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Validation")
    xValidate = data[:, 0:2]
    yValidate = data[:, 2]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    X = generateDataFromGMM(N, gmmParameters)
    return X


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    X = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        X[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    # NOTE TRANPOSE TO GO TO SHAPE (N, n)
    return X.transpose()


def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.show()

def cubic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    phi_X = np.column_stack((
        phi_X, 
    pow(X[:,1],2), X[:,1]*X[:,2], pow(X[:,2],2),
      pow(X[:, 1],3), pow(X[:, 1],2) * X[:, 2], pow(X[:, 2],2) * X[:, 1], pow(X[:,2],3)))
        
    return phi_X


def gradient_descent(loss_func, theta0, X, y, N, *args, **kwargs):
    # Mini-batch GD. Stochastic GD if batch_size=1.

    # Break up data into batches and work out gradient for each batch
    # Move parameters theta in that direction, scaled by the step size.

    # Options for total sweeps over data (max_epochs),
    # and parameters, like learning rate and threshold.

    # Default options
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10

    # Turn the data into batches
    X_batch, y_batch = batchify(X, y, batch_size, N)
    num_batches = len(y_batch)
    print("%d batches of size %d\n" % (num_batches, batch_size))

    theta = theta0
    m_t = np.zeros(theta.shape)

    trace = {}
    trace['loss'] = []
    trace['theta'] = []

    # Main loop:
    for epoch in range(1, max_epoch + 1):
        # print("epoch %d\n" % epoch)
        
        loss_epoch = 0
        for b in range(num_batches):
            X_b = X_batch[b]
            y_b = y_batch[b]
            # print("epoch %d batch %d\n" % (epoch, b))

            # Compute NLL loss and gradient of NLL function
            loss, gradient = loss_func(theta, X_b, y_b, *args)
            loss_epoch += loss
            
            # Steepest descent update
            theta = theta - alpha * gradient
            
            # Terminating Condition is based on how close we are to minimum (gradient = 0)
            if np.linalg.norm(gradient) < epsilon:
                print("Gradient Descent has converged after {} epochs".format(epoch))
                break
                
        # Storing the history of the parameters and loss values per epoch
        trace['loss'].append(np.mean(loss_epoch))
        trace['theta'].append(theta)
        
        # Also break epochs loop
        if np.linalg.norm(gradient) < epsilon:
            break

    return theta, trace




# Breaks the matrix X and vector y into batches
def batchify(X, y, batch_size, N):
    X_batch = []
    y_batch = []

    # Iterate over N in batch_size steps, last batch may be < batch_size
    for i in range(0, N, batch_size):
        nxt = min(i + batch_size, N + 1)
        X_batch.append(X[i:nxt, :])
        y_batch.append(y[i:nxt])

    return X_batch, y_batch



def analytical_solution(X, y, gamma):
    # Analytical solution is (X^T*X)^-1 * X^T * y 
    # Gets (theta)NLL
    aa = X.T.dot(X) 
    return np.linalg.inv(((X-np.mean(X)).T.dot(X) + np.identity(10)*gamma)).dot((X- np.mean(X)).T).dot((y-np.mean(y)))



# Mean Squared Error (MSE) loss
def lin_reg_loss(theta, X, y):
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error**2)

    return loss_f

# Need to provide a function handle to the optimizer, which returns the loss objective, e.g. MSE
def func_mse(theta):
    return lin_reg_loss(theta, x_V, yValidate)


opts = {}
opts['max_epoch'] = 1000
opts['alpha'] = 1e-3
opts['tolerance'] = 1e-3

opts['batch_size'] = 10

xTrain, yTrain, xValidate, yValidate = hw2q2()
theta0 = np.random.randn(10) 

xTrain = np.column_stack((np.ones(len(xTrain)), xTrain))  
x_Validate = np.column_stack((np.ones(len(xValidate)), xValidate))  

x_T = cubic_transformation(xTrain)  
x_V = cubic_transformation(x_Validate)  


fig, ax = plt.subplots(figsize=(10, 10))
fig, ax2 = plt.subplots(figsize=(10, 10))

error_range = np.linspace(pow(10,-4), pow(10,4),10000)
print(len(error_range))

colors = 'bgrcmykw'
ax.scatter(x_V[:, 1], yValidate, s=1)
error = np.ones(len(error_range))
theta = np.ones([len(error_range),10])
for index, power in enumerate(error_range):
    theta_opt = analytical_solution(x_T, yTrain, power)
    theta[index] = theta_opt
    analytical_preds = x_V.dot(theta_opt)
    # Minimize using a default unconstrained minimization optimization algorithm
    mse_model = minimize(func_mse, theta_opt, tol=1e-6)
    # res is the optimization result, has an .x property which is the solution array, e.g. theta*
    error[index] = mse_model.fun
#    ax.scatter(x_T[:, 1], mse_preds, color='red', label="MSE")


print("Min MSE for MAP: ", np.min(error))
print("Optimal Threshold for MAP: ", error_range[np.argmin(error)])
print("Theta for MAP min MSE : ")
print(theta[np.argmin(error)])
analytical_preds = x_V.dot(theta[np.argmin(error)])
# Minimize using a default unconstrained minimization optimization algorithm
mse_model = minimize(func_mse, theta[np.argmin(error)], tol=1e-6)
# res is the optimization result, has an .x property which is the solution array, e.g. theta*
mse_preds = x_T.dot(mse_model.x)
ax2.scatter(error_range, error, color='red', label="Error")

ax.scatter(x_T[:, 1], mse_preds, color='red', label="MSE")

#print("Theta for MLE: ")
#print(mse_model.x)
#print("MSE for MLE: ", mse_model.fun)
# Predictions with our optimized theta
#

# Plot the learned regression line on our original scatter plot


plt.show(block=True)


