# Widget to manipulate plots in Jupyter notebooks


import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

from math import ceil, floor 

import numpy as np
import seaborn as sns
from scipy.stats import norm, multivariate_normal

np.set_printoptions(suppress=True)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)


N_train = [20, 200, 2000]
N_valid = 10000

mu = np.array([[2, 2]])

internal_mu = np.array([
        [3, 0], [0, 3], 
    ])

Sigma = np.array([
                [
                    [1, 0],
                    [0, 1]
                ]
                ])

internal_Sigma = np.array([
    [
    [2, 0],
    [0, 1]
    ],
        [
    [1, 0],
    [0, 2]
    ]
    ])

n = mu.shape[1]
priors = np.array([0.65, 0.35])  
C = len(priors)

n = mu.shape[1]

X_T = [np.zeros([N_train[0], n]), np.zeros([N_train[1], n]), np.zeros([N_train[2], n])]
y_T = [np.zeros(N_train[0]), np.zeros(N_train[1]), np.zeros(N_train[2])]


X_train = []
y_train = []
Ny_train = []

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
t = 0
# Decide randomly which samples will come from each component
for index in range(len(X_T)):
    u = np.random.rand(N_train[index])
    thresholds = np.cumsum(priors)

    ax[floor(t/2), t%2].set_title("Training N={} data".format(N_train[index]))

    for c in range(C):
        c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
        c_N = len(c_ind)  # No. of samples in this component
        y_T[index][c_ind] = c * np.ones(c_N)
        u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
        if(c == 0):
            u = np.random.rand(c_N)
            for num, ind_val in enumerate(c_ind):
                if(u[num] >= 0.5):
                    X_T[index][ind_val] = multivariate_normal.rvs(internal_mu[0], internal_Sigma[0], 1)
                else:
                    X_T[index][ind_val] = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)

        else:
            X_T[index][c_ind] = multivariate_normal.rvs(mu[0], Sigma[0], c_N)

    X_t = np.column_stack((np.ones([N_train[index]]), X_T[index]))
    
    
    n = X_t.shape[1]

    Ny_t = np.array((sum(y_T[index] == 0), sum(y_T[index] == 1)))
    n = X_t.shape[1]
        # Add to lists
    X_train.append(X_t)
    y_train.append(y_T[index])
    Ny_train.append(Ny_t)
    ax[floor(t/2), t%2].scatter(X_t[:, 1], X_t[:, 2], c=y_T[index])
    ax[floor(t/2), t%2].set_ylabel("y-axis")
    ax[floor(t/2), t%2].set_aspect('equal')
    t += 1

X_valid = np.zeros([N_valid, mu.shape[1]])
y_valid = np.zeros(N_valid)

u = np.random.rand(N_valid)
thresholds = np.cumsum(priors)
for c in range(C):
    c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
    c_N = len(c_ind)  # No. of samples in this component
    y_valid[c_ind] = c * np.ones(c_N)
    u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
    if(c == 0):
        u = np.random.rand(c_N)
        for num, ind_val in enumerate(c_ind):
            if(u[num] >= 0.5):
                X_valid[ind_val] = multivariate_normal.rvs(internal_mu[0], internal_Sigma[0], 1)
            else:
                a = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)
                X_valid[ind_val] = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)

    else:
        X_valid[c_ind] = multivariate_normal.rvs(mu[0], Sigma[0], c_N)

X_valid = np.column_stack((np.ones(N_valid), X_valid))  
Ny_valid = np.array((sum(y_valid == 0), sum(y_valid == 1)))

ax[1, 1].set_title("Validation N={} data".format(N_valid))
ax[1, 1].scatter(X_valid[:, 1], X_valid[:, 2], c=y_valid)
ax[1, 1].set_ylabel("y-axis")
ax[1, 1].set_aspect('equal')

x1_valid_lim = (floor(np.min(X_valid[:,1])), ceil(np.max(X_valid[:,1])))
x2_valid_lim = (floor(np.min(X_valid[:,2])), ceil(np.max(X_valid[:,2])))

plt.setp(ax, xlim=x1_valid_lim, ylim=x2_valid_lim)
plt.tight_layout()
plt.show()

fig;

# Define the logistic/sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Define the prediction function y = 1 / (1 + np.exp(-X*theta))
# X.dot(theta) inputs to the sigmoid referred to as logits
def predict_prob(X, theta):
    logits = X.dot(theta)
    return sigmoid(logits)

# NOTE: This implementation may encounter numerical stability issues... 
# Read into the log-sum-exp trick OR use a method like: sklearn.linear_model import LogisticRegression
def log_reg_loss(theta, X, y):
    # Size of batch
    B = X.shape[0]

    # Logistic regression model g(X * theta)
    predictions = predict_prob(X, theta)

    # NLL loss, 1/N sum [y*log(g(X*theta)) + (1-y)*log(1-g(X*theta))]
    error = predictions - y
    nll = -np.mean(y*np.log(predictions) + (1 - y)*np.log(1 - predictions))
    
    # Partial derivative for GD
    g = (1 / B) * X.T.dot(error)
    
    # Logistic regression loss, NLL (binary cross entropy is another interpretation)
    return nll, g


# Options for mini-batch gradient descent
opts = {}
opts['max_epoch'] = 1000
opts['alpha'] = 1e-3
opts['tolerance'] = 1e-3

opts['batch_size'] = 10

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

# Can also use: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
def quadratic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    phi_X = np.column_stack((phi_X, X[:, 1] * X[:, 1], X[:, 1] * X[:, 2], X[:, 2] * X[:, 2]))
        
    return phi_X

# Use the validation set's sample space to bound the grid of inputs
# Work out bounds that span the input feature space (x_1 and x_2)
bounds_X = np.array((floor(np.min(X_valid[:,1])), ceil(np.max(X_valid[:,1]))))
bounds_Y = np.array((floor(np.min(X_valid[:,2])), ceil(np.max(X_valid[:,2]))))

def create_prediction_score_grid(theta, poly_type='Q'):
    # Create coordinate matrices determined by the sample space; can add finer intervals than 100 if desired
    xx, yy = np.meshgrid(np.linspace(bounds_X[0], bounds_X[1], 200), np.linspace(bounds_Y[0], bounds_Y[1], 200))

    # Augment grid space with bias ones vector and basis expansion if necessary
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_aug = np.column_stack((np.ones(200*200), grid)) 
    if poly_type == 'Q':
        grid_aug = quadratic_transformation(grid_aug)

    # Z matrix are the predictions resulting from sigmoid on the provided model parameters
    Z = predict_prob(grid_aug, theta).reshape(xx.shape)
    
    return xx, yy, Z


def plot_prediction_contours(X, theta, ax, poly_type='Q'):
    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    # Once reshaped as a grid, plot contour of probabilities per input feature (ignoring bias)
    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.55)
    ax.set_xlim([bounds_X[0], bounds_X[1]])
    ax.set_ylim([bounds_Y[0], bounds_Y[1]])

    
def plot_decision_boundaries(X, labels, theta, ax, poly_type='Q'): 
    # Plots original class labels and decision boundaries
    ax.plot(X[labels==0, 1], X[labels==0, 2], 'o', label="Class 0")
    ax.plot(X[labels==1, 1], X[labels==1, 2], '+', label="Class 1")
    
    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    # Once reshaped as a grid, plot contour of probabilities per input feature (ignoring bias)
    cs = ax.contour(xx, yy, Z, levels=1, colors='k')

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect('equal')

    
def report_logistic_classifier_results(X, theta, labels, N_labels, ax, poly_type='Q'):
    """
    Report the probability of error and plot the classified data, plus predicted 
    decision contours of the logistic classifier applied to the data given.
    """
    
    predictions = predict_prob(X, theta)  
    # Predicted decisions based on the default 0.5 threshold (higher probability mass on one side or the other)
    decisions = np.array(predictions >= 0.5)
    
    # True Negative Probability Rate
    ind_00 = np.argwhere((decisions == 0) & (labels == 0))
    tnr = len(ind_00) / N_labels[0]
    # False Positive Probability Rate
    ind_10 = np.argwhere((decisions == 1) & (labels == 0))
    fpr = len(ind_10) / N_labels[0]
    # False Negative Probability Rate
    ind_01 = np.argwhere((decisions == 0) & (labels == 1))
    fnr = len(ind_01) / N_labels[1]
    # True Positive Probability Rate
    ind_11 = np.argwhere((decisions == 1) & (labels == 1))
    tpr = len(ind_11) / N_labels[1]

    prob_error = fpr*priors[0] + fnr*priors[1]

    print("The total error achieved with this classifier is {:.3f}".format(prob_error))
    
    # Plot all decisions (green = correct, red = incorrect)
    ax.plot(X[ind_00, 1], X[ind_00, 2], 'og', label="Class 0 Correct", alpha=.25)
    ax.plot(X[ind_10, 1], X[ind_10, 2], 'or', label="Class 0 Wrong")
    ax.plot(X[ind_01, 1], X[ind_01, 2], '+r', label="Class 1 Wrong")
    ax.plot(X[ind_11, 1], X[ind_11, 2], '+g', label="Class 1 Correct", alpha=.25)

    # Draw the decision boundary based on whether its linear (L) or quadratic (Q)
    plot_prediction_contours(X, theta, ax, poly_type)
    ax.set_aspect('equal')


# NOTE that the initial parameters have added dimensionality to match the basis expansion set
theta0_quadratic = np.random.randn(n+3)

fig_decision, ax_decision = plt.subplots(3, 2, figsize=(15, 15));

print("Training the logistic-quadratic model with GD per data subset"),
for i in range(len(N_train)):
    shuffled_indices = np.random.permutation(N_train[i]) 
    
    # Shuffle row-wise X (i.e. across training examples) and labels using same permuted order
    X = X_train[i][shuffled_indices]
    y = y_train[i][shuffled_indices]
    
    # Important transformation line to add monic polynomial terms for a quadratic
    X_quad = quadratic_transformation(X)
    theta_gd, trace = gradient_descent(log_reg_loss, theta0_quadratic, X_quad, y, N_train[i], **opts)

    print("Logistic-Quadratic N={} GD Theta: {}".format(N_train[i], theta_gd))
    print("Logistic-Quadratic N={} NLL: {}".format(N_train[i], trace['loss'][-1]))

    # Convert our trace of parameter and loss function values into NumPy "history" arrays:
    theta_hist = np.asarray(trace['theta'])
    nll_hist = np.array(trace['loss'])
    
    plot_decision_boundaries(X_quad, y, theta_gd, ax_decision[i, 0], poly_type='Q')
    ax_decision[i, 0].set_title("Decision Boundary for \n Logistic-Quadratic Model N={}".format(X.shape[0]))

    # Quadratic: use validation data (10k samples) and make decisions in report results routine
    X_valid_quad = quadratic_transformation(X_valid)
    report_logistic_classifier_results(X_valid_quad, theta_gd, y_valid, Ny_valid, ax_decision[i, 1], poly_type='Q')
    ax_decision[i, 1].set_title("Classifier Decisions on Validation Set \n Logistic-Quadratic Model N={}".format(N_train[i]))


# Again use the most sampled subset (validation) to define x-y limits
plt.setp(ax_decision, xlim=x1_valid_lim, ylim=x2_valid_lim)

# Adjust subplot positions
plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.6, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.3)

# Super plot the legends
handles, labels = ax_decision[0, 1].get_legend_handles_labels()
fig_decision.legend(handles, labels, loc='lower center')

plt.show()