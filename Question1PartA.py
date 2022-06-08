import matplotlib.pyplot as plt # For general plotting

import numpy as np
import random
import math

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)


random.seed(7)


# TAKEN FROM MARK ZOLOTAS
plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

N = [20, 200, 2000, 10000]

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

X = [np.zeros([N[0], n]), np.zeros([N[1], n]), np.zeros([N[2], n]), np.zeros([N[3], n])]
y = [np.zeros(N[0]), np.zeros(N[1]), np.zeros(N[2]), np.zeros(N[3])]

internal_label = [np.zeros([N[0],1]), np.zeros([N[1],1]), np.zeros([N[2],1]), np.zeros([N[3],1])]
# Decide randomly which samples will come from each component
for index in range(len(X)):
    u = np.random.rand(N[index])
    thresholds = np.cumsum(priors)
    for c in range(C):
        c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
        c_N = len(c_ind)  # No. of samples in this component
        y[index][c_ind] = c * np.ones(c_N)
        u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
        if(c == 0):
            u = np.random.rand(c_N)
            for num, ind_val in enumerate(c_ind):
                if(u[num] < 0.5):
                    X[index][ind_val] = multivariate_normal.rvs(internal_mu[0], internal_Sigma[0], 1)
                    internal_label[index][ind_val] = 0
                else:
                    X[index][ind_val] = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)
                    internal_label[index][ind_val] = 0

        else:
            X[index][c_ind] = multivariate_normal.rvs(mu[0], Sigma[0], c_N)


ind = 3
# Plot the original data and their true labels
plt.figure(figsize=(12, 10))
plt.plot(X[ind][y[ind]==0, 0], X[ind][y[ind]==0, 1], 'bo', label="Class 0")
plt.plot(X[ind][y[ind]==1, 0], X[ind][y[ind]==1, 1], 'rx', label="Class 1");


plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Labels")
plt.tight_layout()
plt.show()

Y = np.array(range(C))
Lambda = np.ones((C, C)) - np.identity(C)


internal = internal = round(random.random())
m00 = multivariate_normal.pdf(X[3], internal_mu[0], internal_Sigma[0])
m01 = multivariate_normal.pdf(X[3], internal_mu[1], internal_Sigma[1])
m2 = multivariate_normal.pdf(X[3], mu[0], Sigma[0])
class_priors = np.diag(priors)



discriminant_score_erm = m2 / (m00*0.5 + m01*0.5)  


# Decision Rule
# As of right now, it selects the index that has the lowest value for the specific column
# Value is R(D(x) = i | x)
gamma_map = priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]

decisions = discriminant_score_erm >= gamma_map

fig = plt.figure(figsize=(12, 10))
marker_shapes = 'ox+*.' # Accomodates up to C=5
marker_colors = 'brgm'

# Get sample class counts
sample_class_counts = np.array([sum(y[3] == j) for j in Y])

# Confusion matrix

conf_mat = np.zeros((C, C))
display_mat = np.zeros((C, C))
for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions==i) & (y[3]==j))
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j] # Average over class sample count
        display_mat[i,j] =  len(ind_ij)

        # True label = Marker shape; Decision = Marker Color
        marker = marker_shapes[j] + marker_colors[i]
        plt.plot(X[3][ind_ij, 0], X[3][ind_ij, 1], marker)

        if i != j:
            plt.plot(X[3][ind_ij, 0], X[3][ind_ij, 1], marker, markersize=16)
            
#Confusion Matrix
# TP | FN
# FP |
print("Confusion Matrix (rows: Predicted class, columns: True class):")
print(display_mat)

print("Confusion matrix by average:")
print(conf_mat)

correct_class_samples = np.sum(np.diag(display_mat))
print("Total Mumber of Misclassified Samples: {:.4f}".format(N[3] - correct_class_samples))

prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N[3])
print("Minimum Probability of Error: ", prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()


# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

Nl = np.array([sum(y[3] == l) for l in Y])

# True Negative Probability
ind_00_map = np.argwhere((decisions==0) & (y[3]==0))
p_00_map = len(ind_00_map) / Nl[0]

# False Positive Probability
ind_0x_map = np.argwhere((decisions==0) & (y[3]!=0))
p_0x_map = len(ind_0x_map) / Nl[0]
# False Negative Probability
ind_11_map = np.argwhere((decisions==1) & (y[3]==1))
p_11_map = len(ind_11_map) / Nl[1]
# True Positive Probability
ind_1x_map = np.argwhere((decisions==1) & (y[3]!=1))
p_1x_map = len(ind_1x_map) / Nl[1]

X_Train = X[:3]

X = X[3]
y = y[3]
Nl = np.array([sum(y == l) for l in Y])
N = N[3]

decisions_map = discriminant_score_erm >= gamma_map

# True Negative Probability
ind_00_map = np.argwhere((decisions_map==0) & (y==0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map==1) & (y==0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map==0) & (y==1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map==1) & (y==1))
p_11_map = len(ind_11_map) / Nl[1]

# Probability of error for MAP classifier, empirically estimated
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
print("Smallest P(error) for ERM = {}".format(prob_error_erm))

# Display MAP decisions
plt.ioff() # These are Jupyter only lines to avoid showing the figure when I don't want
fig_disc_grid, ax_disc = plt.subplots(figsize=(10, 10));
plt.ion() # Re-activate "interactive" mode

# class 0 circle, class 1 +, correct green, incorrect red
ax_disc.plot(X[ind_00_map, 0], X[ind_00_map, 1], 'og', label="Correct Class 0");
ax_disc.plot(X[ind_10_map, 0], X[ind_10_map, 1], 'or', label="Incorrect Class 0");
ax_disc.plot(X[ind_01_map, 0], X[ind_01_map, 1], '+r', label="Incorrect Class 1");
ax_disc.plot(X[ind_11_map, 0], X[ind_11_map, 1], '+g', label="Correct Class 1");

ax_disc.legend();
ax_disc.set_xlabel(r"$x_1$");
ax_disc.set_ylabel(r"$x_2$");
ax_disc.set_title("MAP Decisions (RED incorrect)");
fig_disc_grid.tight_layout();


# TAKEN FROM MARK ZOLOTAS
from sys import float_info # Threshold smallest positive floating value

# TAKEN FROM MARK ZOLOTAS
# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] + 
             sorted_score +
             [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]
    ind01 = [np.argwhere((d==0) & (label==1)) for d in decisions]
    p01 = [len(inds)/Nlabels[1] for inds in ind01]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11, p01))


    

    return roc, np.array(taus)

# Construct the ROC for ERM by changing log(gamma)
roc_erm, taus = estimate_roc(discriminant_score_erm, y)
roc_map = np.array((p_10_map, p_11_map))

prob_error = np.array((roc_erm[0,:],1- roc_erm[1,:])).T.dot(Nl.T/N)


min_prob_error = np.min(prob_error)
min_ind = np.argmin(prob_error)

print("Index of Minumum Probability Error: ", min_ind)

print("Empirical Estimated Probability of Error: {:.4f}".format(min_prob_error))
print("Theoretical Estimated Probability of Error: {:.4f}".format(prob_error_erm))
print("Theoretical Threshold: {:.4f}".format(gamma_map))
print("Empirical Threshold: {:.4f}".format(math.exp(taus[min_ind])))


# TAKEN FROM MARK ZOLOTAS
fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
ax_roc.plot(roc_erm[0,min_ind], roc_erm[1,min_ind], 'ro',label="Empirical Minimum P(Error) MAP", markersize=16)

ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.grid(True)

plt.show(block=True)


