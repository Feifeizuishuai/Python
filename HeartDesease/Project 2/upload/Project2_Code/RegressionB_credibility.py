import matplotlib.pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from scipy import stats
import torch

mat_data = loadmat('heart_2.mat')
X = mat_data['X']
y = mat_data['Y'].squeeze()
y_ann = mat_data['Y']
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape

# Add offset attribute
X_lr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,10))

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

#------------------------ann--------------------------------

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
optimial_h_outer = [] # make a list for storing generalizaition error in each loop
optimial_h_outer_error = []
#--------------------------------------------------------------------------

k=0
for train_index, test_index in CV.split(X_lr,y):
    
    # extract training and test set for current CV fold
    X_train = X_lr[train_index]
    y_train = y[train_index]
    X_test = X_lr[test_index]
    y_test = y[test_index]
    
    X_train_ann = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train_ann = torch.tensor(y_ann[train_index], dtype=torch.float)
    X_test_ann = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test_ann = torch.tensor(y_ann[test_index], dtype=torch.uint8)
    
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    x_lan=0
    for landa in lambdas:
         lambdaI = landa * np.eye(M)

         w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
         Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
         Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
         print('- E_test: {0}'.format(Error_test_rlr.sum()/10))
         print('lambda {0}'.format(Error_test_rlr.min()))

    
         x_lan+=1

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    print('- E_test_baseline: {0}'.format(Error_test.sum()/10))
    


    
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        
#        figure(k)
#        subplot(1,2,1)
#        plt(range(1,len(Error_train_rlr)), Error_train_rlr[1:])
#        xlabel('Iteration')
#        ylabel('Squared error (crossvalidation)')    
        
#        subplot(1,3,3)
#        bmplt(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
#        clim(-1.5,0)
#        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    
    #----------------------------------ann---------------------------------------
    #----------------------------------------------------------------------------
    j=0   
    for train_index, test_index in CV.split(X_train_ann,y_train_ann):
        print('Inner loop'.format(K))
        
        # extract training and test set for current CV fold
        X_train_2 = X[train_index,:]
        y_train_2 = y_ann[train_index]
        X_test_2 = X[test_index,:]
        y_test_2 = y_ann[test_index]

        X_train_2n = torch.tensor(X[train_index,:], dtype=torch.float)
        y_train_2n = torch.tensor(y_ann[train_index], dtype=torch.float)
        X_test_2n = torch.tensor(X[test_index,:], dtype=torch.float)
        y_test_2n = torch.tensor(y_ann[test_index], dtype=torch.uint8)
        
        for n_hidden_units in range(10):
            print(n_hidden_units)
           #Define the model, see also Exercise 8.2.2-script for more information.
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M-1, n_hidden_units+1), #M features to H hiden units
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
    
    
    
    
    #------------------------------------------------------------------------------
    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

Error_ann = np.asarray(optimial_h_outer_error)


#------------------------different comparisons with different z-----------------------------
#z = (Error_ann.reshape(-1)-Error_test.reshape(-1))
#z = (Error_test.reshape(-1)-Error_test_rlr.reshape(-1))
z = (Error_ann.reshape(-1)-Error_test_rlr.reshape(-1))
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
x=np.linspace(-2.5,-0.5,300)
y=stats.norm.pdf(x,zb,sig)
plt.plot(x,y,lw=2,color='black')
plt.title('ANN vs Linear Regression')
plt.vlines(zL,0,1.8,linestyles='--',colors='r')
plt.text(zL+0.1,0.18, "95% Left",color='r')
plt.vlines(zH,1.8,0,linestyles='--',colors='b')
plt.text(zH+0.1,0.18, "95% Right",color='b')
plt.grid()
plt.show()
