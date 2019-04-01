from IPython.display import display
import pandas as pd
def desc_nan(df):
    """
    Show a table that describe about all columns of @df that contain NaN.
    """
    row, col = df.shape
    nas = row - df.describe(include='all').T['count']
    nas = nas[nas > 0]
    ps = nas / row * 100
    ms = df.mode().iloc[0]
    nadf = pd.DataFrame(data={
        'Na Count': nas,
        'Na Percentage': ps,
        'mode': ms
    }).loc[nas.index]
    dnadf = df.describe().T.reindex(nadf.index)
    nadf = pd.concat([nadf, dnadf.iloc[:, 1:]], axis=1)
    display(nadf)

#%%

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def plot_ft_dist(df, fig_height=4, no_of_col=2,verbose=True):
    """
    Plot distribution of all numeric columns in df

    Parameters
    ----------
    df: pandas.DataFrame

    fig_height: int

    no_of_col: int
    """
    df = df.copy()
    # prepare df
    cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
    if verbose:
        print("Ignored categorical columns: ", cat_cols)
        print("")
    df_hasnan = df.isna().any().any() # if any row (1st any) in any column (2nd any) has NaN
    if df_hasnan:
        nan_cols = list(df.loc[:,df.isna().any()].columns)
        df.dropna(inplace=True, axis=1)
        if verbose:
            print("dropped NaN cols:", str(nan_cols))
            print("")
    try:    
        desc = df.describe().T
        std0 = list(desc.loc[desc['std']==0].index)
        if verbose:
            print("dropped std=0 cols:", str(std0))
            print("")
        for col in std0:
            del df[col]
    except KeyError:
        pass
    
    idx = df.dtypes[df.dtypes != 'object'].index
    # prepare frame
    f, axes = plt.subplots(
        int(np.ceil(len(idx) / no_of_col)),
        no_of_col,
        figsize=(5 * no_of_col, np.ceil(len(idx) / no_of_col) * fig_height))
    sns.set(style="white", palette="muted", color_codes=True)
    n = 0
    for i in idx:
        sns.distplot(
            df[i],
            color='b',
            hist=True,
            kde_kws={"shade": True},
            ax=axes[n // no_of_col, n % no_of_col])
        n += 1
    plt.show()

#%%

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def plot_ft_ix(df, fig_height=4, no_of_col=2,verbose=True):
    """
    Plot all data values of columns in df by index

    Parameters
    ----------
    df: pandas.DataFrame, data

    fig_height: int

    no_of_col: int
    """
    df = df.copy()
    # prepare df
    cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
    if verbose:
        print("Ignored categorical columns: ", cat_cols)
        print("")
    df_hasnan = df.isna().any().any()
    if df_hasnan:
        nan_cols = list(df.loc[:,df.isna().all()].columns) # columns is nan - columns have no value
        df.dropna(inplace=True, axis=1)  # drop nan columns
        if verbose:
            print("dropped NaN cols:", str(nan_cols))
            print("")
    try:    
        desc = df.describe().T
        std0 = list(desc.loc[desc['std']==0].index)
        if verbose:
            print("dropped std=0 cols:", str(std0))
            print("")
        for col in std0:
            del df[col]
    except KeyError:
        pass
    
    idx = df.dtypes[df.dtypes != 'object'].index
     # prepare frame
    f, axes = plt.subplots(
        int(np.ceil(len(idx) / no_of_col)),
        no_of_col,
        figsize=(5 * no_of_col, np.ceil(len(idx) / no_of_col) * fig_height))
    sns.set(style="white", palette="muted", color_codes=True)
    n = 0
    for i in idx:
        ax=axes[n // no_of_col, n % no_of_col]
        ax.plot(df.index, df[i]) # plot data by index
        ax.set_title(i)
        n += 1
   
    plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
    
def plot_reg_result(y_test, y_pred):

    x = y_test.values
    y = y_pred

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))
    binwidth = (xymax - xymin)/10
    ceil = (int(xymax/binwidth) + 2 ) * binwidth
    floor = (int(xymin/binwidth) - 2 ) * binwidth

    axScatter.set_xlim((floor, ceil))
    axScatter.set_ylim((floor, ceil))

    p = np.linspace(floor,ceil)
    axScatter.plot(p, p,"-r")

    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.grid(True)
    plt.show()

#%%
    
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
def plot_confusion_matrix(y_test, y_pred):
    """
    y_test: dataframe
    y_pred: numpy array
    """
    test_labels = set(y_test.iloc[:,0])    
    pred_labels = set(y_pred)
    labels = sorted(list(test_labels | pred_labels)) # union of sets
    cm = confusion_matrix(y_test, y_pred,labels=labels)
    cm = np.rot90(cm,1) 
    sns.set(style="ticks")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap='Greens', interpolation='none')
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    # unknown intention
#     ax.xaxis.set_major_locator(plt.MaxNLocator(len(labels)))
#     ax.yaxis.set_major_locator(plt.MaxNLocator(len(labels)))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels[::-1])
    thresh = cm.max()*0.8    #threshold to switch the color of the text inside the confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # old
#     plt.xlabel('Predicted values')
#     plt.ylabel('True values')
    # new
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.tight_layout()
    plt.show()