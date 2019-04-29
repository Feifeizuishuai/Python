# exercise 8.2.5
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

# Load Matlab data file and extract variables of interest
mat_data = loadmat('heart_2.mat')
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
X = mat_data['X']
y = mat_data['Y']
K = 10                   # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)
N, M = X.shape
C = 2
# Normalize data
X = stats.zscore(X);
# Parameters for neural network classifier
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 50       # stop criterion 2 (max epochs in training)
# K-fold crossvalidation
mse = np.empty((10,1))
optimal_h_inner = np.empty((K,1))
optimal_h_inner_error = np.empty((K,1))
errors  = np.empty((10,1)) # make a list for storing generalizaition error in each loop
# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
loss_fn = torch.nn.MSELoss()

#print('Training model of type:\n\n{}\n'.format(str(model())))
optimial_h_outer = [] # make a list for storing generalizaition error in each loop
optimial_h_outer_error = []
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('Crossvalidation fold: {0}/{1}\n'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.uint8)
    
    j=0   
    for train_index, test_index in CV.split(X_train,y_train):
        print('Inner loop'.format(K))
        
        # extract training and test set for current CV fold
        X_train_2 = X[train_index,:]
        y_train_2 = y[train_index]
        X_test_2 = X[test_index,:]
        y_test_2 = y[test_index]
    
       
#        y_train_2n = StandardScaler().fit_transform(y_train_2.reshape(-1,1)).reshape(-1)
#        y_test_2n =  StandardScaler().fit_transform(y_test_2.reshape(-1,1)).reshape(-1)
        
        X_train_2n = torch.tensor(X[train_index,:], dtype=torch.float)
        y_train_2n = torch.tensor(y[train_index], dtype=torch.float)
        X_test_2n = torch.tensor(X[test_index,:], dtype=torch.float)
        y_test_2n = torch.tensor(y[test_index], dtype=torch.uint8)
        

        
        for n_hidden_units in range(10):
            print(n_hidden_units)
           #Define the model, see also Exercise 8.2.2-script for more information.
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units+1), #M features to H hiden units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units+1, 1), # H hidden units to 1 output neuron
                            torch.nn.Sigmoid() # final tranfer function
                            )
            
    
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_2n,
                                                               y=y_train_2n,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)


            # Determine estimated class labels for test set
            y_test_est = net(X_test_2n)
            #y_test_est = y_sigmoid>.5
            se = (y_test_est.float()-y_test_2n.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test_2n)).data.numpy() #mean
            errors[n_hidden_units]=mse # store error rate for current CV fold 
            # Determine errors and errors
            #e = y_test_est != y_test_2n
#            error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
#            errors.append(error_rate) # store error rate for current CV fold 
#            se = (y_test_est-y_test_2n)**2 # squared error
#            mse[n_hidden_units] = (sum(se).type(torch.float)/len(y_test_2n)).data.numpy() #mean
#            mse[n_hidden_units] = np.square(y_test_est-y_test_2n).sum()/y_test.shape[0]
        
            
        
        optimal_h_inner_error[j] = float(errors.min())
        optimal_h_inner[j] = int(np.argmin(errors) + 1)
        print(optimal_h_inner)    
    #print('\n\tBest loss: {}\n'.format(final_loss))
        j+=1 
    #h**
    optimial_h_outer.append( optimal_h_inner[np.argmin(optimal_h_inner_error)])
    optimial_h_outer_error.append(min(optimal_h_inner_error))
    print(optimial_h_outer)
       
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve)
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)))
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))
