import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as scs
from scipy.stats import norm
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

import pymc3 as pm
import arviz as az

import warnings
warnings.filterwarnings("ignore")

sns.set(style="darkgrid")
import matplotlib.gridspec as gridspec
import seabornfig2grid as sfg

#### FUNCTIONS FOR EDA ####



def target_distribution(target, save =False, trace_name = None):
    """
    Plot Histogram and boxplot for target variable
    params: target - Series, np.array
    """
    fig = plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.ylabel('Frequency')
    sns.distplot(target, ) #fit = norm
    plt.subplot(2,1,2)
    plt.boxplot(target, vert = False)
    plt.xlabel('Depended variable')
    
    if save:
        plt.savefig(trace_name)
    
    return plt.show()


def plot_relationship(df, y):
    """
    Build plot to show relationship btween df values and y values
    params: 
            df - pd.Dataframe, numeric values
            y  - target variable name, string 
    """
    ncols=int(len(df.columns)/2)
    nrows=int(ncols/2)
    
    joinplots = []

    for col in df.columns[:-1]:
        jp_col = sns.jointplot(x= df[col], y= df[y], kind='reg', space=0, color='steelblue')
        joinplots.append(jp_col)

    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(nrows, ncols)

    for i,jp in enumerate(joinplots):
        mg= sfg.SeabornFig2Grid(jp, fig, gs[i])

    gs.tight_layout(fig)
    gs.update(top=0.9)

    plt.show()


def features_corr_matrix(df, threshold, matrix = False):
    
    """
    Plot corr matrix of btw df columns
    
    params: 
            df - dataframe
            threshold - critical value for feature selection
    return: 
            plot, list of sugested features to exclude
    """
    # Create correlation matrix    
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
   
    plt.show()
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if matrix:
        
        corr_matrix = df.corr()[df.corr()>threshold]
        corr_columns = []
        for col,v in corr_matrix.iteritems():
            if v.sum()>threshold+1:
                corr_columns.append(col)
        
        print('Based on threshold {} , sugested featrures to drop - {}'.format(threshold,corr_columns))
        return corr_matrix[corr_columns].dropna()
        
        
    if len(to_drop)==0:
        print('Based on threshold {} , there is no highly corelated features'.format(threshold))
    else :
        print('Based on threshold {} , sugested featrures to drop - {}'.format(threshold,to_drop))
    
    
    
# def plot_relationship(data, target):
    
#     for col in [col for col in data.columns if col not in [target]]:
#         sns.jointplot(x= col, y= target, data= data, kind='reg', height=4.5, )
#         plt.xlabel(col, size = 5); plt.ylabel(target, size = 5); 
#         plt.show()


#### FUNCTIONS FOR MODELING MULTIPLE LINEAR REGRESSION ####


def train_test_valid(df_features, target):
    """
    Function to create Train and test split
    Fit linear regression to the data 
    Evaluate the model 
    
    params : df_features - Data frame with features
             target - series with numeric values !!!! target should be np.log(target)
    """
    #call train_test_split on the data and capture the results # randomles is consistent
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, random_state=9,test_size=0.2)
    
    #instantiate a linear regression object
    lm = linear_model.LinearRegression()
    
    #fit the linear regression to the data
    lm = lm.fit(X_train, y_train)
    
    y_train_pred = lm.predict(X_train)
    
    # we have the predictions, we need to exponentiate them to get them back into the original scale, dollars
    #y_train_pred = np.exp(y_train_pred)
    
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    #print('Root Mean Squared Error:' , train_rmse)
    
    #Predicting the Test Set
    y_pred = lm.predict(X_test)
    #y_pred = np.exp(y_pred)
    
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    #print('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    
    #Comparing our Model's performance on training data versus test data
    print('Training: RMSE', int(train_rmse), "vs. Testing: RMSE", int(test_rmse))
    print('Perfomance : {} %'.format(round(abs((test_rmse-train_rmse)/train_rmse)*100)))
    

def scale_transform_validate(df, target, residuals = False, selection = False):
    
    """
    Scale features and Evaluate model
    params : df_features - Data frame with features (numeric)
             target - series with numeric values!  
             residuals = False. Plot residuals if True 
    """
    
    X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=9, test_size=0.2)
    scaler = StandardScaler()
    
    # fit the scaler to the training data
    scaler.fit(X_train)
    
    #transform the training data
    scaled_data = scaler.transform(X_train)
    
    # create DF
    X_train_scaled = pd.DataFrame(data=scaled_data, columns=df.columns, index=X_train.index)
    
    #transform the test data
    scaled_test_data = scaler.transform(X_test)
    
    #create dataframe
    X_test_scaled = pd.DataFrame(data=scaled_test_data, columns=df.columns, index=X_test.index)
    
    # Fit the model to the training data.
    lm = LinearRegression()
    lm = lm.fit(X_train_scaled, y_train)
    
    # Use the model to predict on the training set and the test set.
    y_train_pred = lm.predict(X_train_scaled)
    y_test_pred = lm.predict(X_test_scaled)
    
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    
    if residuals == True:
        sns.residplot( y_test, y_test_pred, lowess =True, color='g')
    
    if selection == True:
        
        return round(train_rmse), round(test_rmse)
    
    #Comparing our Model's performance on training data versus test data
    else:
        print('Training: RMSE', int(train_rmse), "vs. Testing: RMSE", int(test_rmse))
        print('Perfomance : {} %'.format(round(abs((test_rmse-train_rmse)/train_rmse)*100)))

    return plt.show()


def check_rhat(trace, threshold = 1.01):
    results = az.summary(trace, round_to="none").r_hat.values
    
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(results)
    ax.axvline(threshold, c = 'r')
    
    above = []
    for i,v in enumerate(results):
        if v > threshold:
            above.append(i)
            
    return print('The following parameters have an Rhat greater {}:'.format(threshold, above))



def plot_params_correlation(trace, var_names, y):
    """
    Build plot to show relationship btween df values and y values
    params: 
            df - pd.Dataframe, numeric values
            y  - target variable name, string 
    """
    ncols=round(len(var_names)/2)
    nrows=round(ncols/2)
    
    joinplots = []

    for var in var_names:
        jp_var = sns.jointplot(x= trace[y], y= trace[var], kind='scatter', space=0, color='steelblue')
        jp_var.set_axis_labels(var, y)
        #jp_var = sns.regplot(x= trace[var], y= trace[y], color='steelblue')
        joinplots.append(jp_var)

    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(nrows, ncols)

    for i,jp in enumerate(joinplots):
        mg= sfg.SeabornFig2Grid(jp, fig, gs[i])

    gs.tight_layout(fig)
    gs.update(top=0.9)

    plt.show()

    
##################################################


def plot_correlation(trace, data):
    """
    ploting correlation btw slope and intercept
    params : trace : posterior trace-object from model
             data  : data frame with features and target variable 
    returns : subplots
    """
    
    df = pd.DataFrame(trace['beta'], columns=data.columns[:-1])
    df['alpha'] = trace['alpha']
    
    ncols=round(len(df.columns)/2)
    nrows=round(ncols/3)
    k=0

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))
    fig.suptitle('Correlation Between Slope and Intercept', size = 25, y=0.95)

    for i,col in enumerate(df.columns):
        if i < ncols:
            j=0
            sns.scatterplot('alpha', col, data=df, ax=axes[j,i]);
        else:
            j=1
            sns.scatterplot('alpha', col, data=df, ax=axes[j,k]);
            k+=1
            if k == ncols: break
            
    return plt.show()


###################################

def calculate_contribution(trace, df, plot = True,  summary = False, standardized = False, glm = False):
    
    
    var_names = trace.varnames
    if glm:
        df_summary = az.summary(trace, var_names = df.columns[:-1])[['mean','sd']]
    else:
        df_summary = az.summary(trace, var_names = 'beta')[['mean','sd']]
    df_summary['mean2'] = df_summary['mean'].apply(lambda x: x*x)
    
    if standardized:
        df_summary['sd2'] = 1
    else: 
        df_summary['sd2'] = df_summary['sd'].apply(lambda x: x*x)
    df_summary['mean2*sd2'] =  df_summary.mean2*df_summary.sd2
    
    pcts = list(zip(df_summary.index, 
                [round(value/sum(df_summary['mean2*sd2'])*100, 
                       5) for value in df_summary['mean2*sd2'].values]))
    
    df_summary['contribution_pct'] = [x[1] for x in pcts]
    
    if plot:
        
        #Seaborn Horizontal barplot
        sns.set_style("darkgrid")
        bar,ax = plt.subplots(figsize=(8,4))
        ax = sns.barplot(x="contribution_pct", y=df_summary.index, data=df_summary, ci=None, palette="muted",orient='h' )
        ax.set_title("Contribution to tune_in_i", fontsize=15)
        ax.set_xlabel ("Percentage")
        ax.set_ylabel ("Channel Type")
        for rect in ax.patches:
            ax.text (rect.get_width(), rect.get_y() + rect.get_height() / 2,"%.1f%%"% rect.get_width(), weight='bold' )

        # bar.savefig("Seaborn_Bar_Vertical.png");
    if summary:
        return df_summary
    
    
    # function to check posterior pred check

def ppc_check(model, trace, data, plot_ppc = True, glm= False, predictions= False, samples_n = 200, color = 'C2'):
    
    ppc_zero = pm.sample_posterior_predictive(trace,
                                              samples = samples_n,
                                              model = model 
                                             )
    if glm:
        r2 = az.r2_score(data.TOTAL_CONV.values, ppc_zero['y'])
    else:
        r2 = az.r2_score(data.TOTAL_CONV.values, ppc_zero['conv'])
    
    data_ppc_zero = az.from_pymc3(trace= trace, posterior_predictive= ppc_zero)
    
    if plot_ppc:
        ax = az.plot_ppc(data_ppc_zero, figsize=(10,6), mean= True, color = color,)
        ax.legend(fontsize= 13)
        # ax.title.set_text('Model with psi_.0' + str(model_trace_list.index((model,trace))+1))
        ax.set_xlabel('TOTAL_CONV')
        plt.show()
    else: pass
    
    if glm:
        print('R2 for Model is : {}'.format(round(r2['r2'],3)))
    else:
        print('R2 for Model is : {}'.format(round(r2['r2'],3)))
    
    if predictions:
        return ppc_zero
    else: pass
    