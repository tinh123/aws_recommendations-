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

import pandas as pd
import numpy as np
from datetime import datetime as dt

def cycle_spliter(dataframe, sort = False, to_files=False, padding = False,
                  min_duration=0):
    """
    Params
    ------
    dataframe: pandas.DataFrame
        Input data, must have "Machine_Mode", "Time_stamp" columns
    sort: boolean
        sort dataframe by Time_stamp
    to_files: boolean
        save cycle-based splited dataframe into csv files, name is df_[order]
    padding: boolean
        True: include one previous row before cycle start and 
        one following row after cycle end
    min_duration: integer
        minimum cycle duration (in minutes) to return
    Return: list of DataFrames of cycles
    """
    # separate local object from passed reference object
    df = dataframe.copy()
    
    df["Time_stamp"] = pd.to_datetime(df["Time_stamp"])

    # sometimes, tables are not in Time_stamp order
    if sort: 
        df.sort_values(by="Time_stamp", inplace=True)
    
    # Convert Machine_Mode to numbers. Object-types will be NaN
    df['Machine_Mode'] = pd.to_numeric(df.Machine_Mode, errors="coerce")
    
    # Drop NaN rows in Machine_mode
    if df.Machine_Mode.hasnans:
        df.drop(index=df.loc[df.Machine_Mode.isna()].index,axis=0, inplace=True)

    if "Cycle_Start_Time" not in df.columns:
        df.insert(2,"Cycle_Start_Time",np.nan)
    if "Cycle_End_Time" not in df.columns:
        df.insert(3,"Cycle_End_Time",np.nan)
        
    df.reset_index(drop=True,inplace=True)

    start_idx, start_time = 0, 0
    end_idx, end_time = 0, 0
    count, count_drop = 0, 0
    list_cycles=[]
    for i, this_mode in enumerate(df.Machine_Mode):
        
        try:
            next_mode = df.Machine_Mode[i+1]
        except KeyError: # end of df, stop loop
            break
            
        # define start time
        if this_mode == 0 and next_mode == 3:
            if padding:
                start_idx = i # val = 0
            else:
                start_idx = i+1 # val = 3
            start_time = df.Time_stamp[start_idx]
            
        # define end time
        elif this_mode != 5 and next_mode == 5 and start_idx != None:
            if padding:
                end_idx = i + 1 # val == 5
            else:
                end_idx = i # val != 5
            end_time = df.Time_stamp[end_idx]
            
            # only cycles last >= min_duration are kept
            duration = round((end_time - start_time).total_seconds()/60,1)
            if duration >= min_duration:
                df.loc[start_idx:end_idx, ['Cycle_Start_Time','Cycle_End_Time']] = [start_time, end_time]
                cycle = df.loc[start_idx:end_idx]
                cycle.reset_index(drop=True,inplace=True)
                list_cycles.append(cycle)

                print("start time: {} \t end time: {} \t duration: {} mins".format(
                    start_time,end_time, duration))

                if to_files:
                    exec("df[start_idx:end_idx+1].to_csv('df_{count}.csv', index=False)".format(count=count))
                count += 1
            else:
                count_drop += 1
                
            # reset values for new cycles    
            start_idx, start_time = None, None
            end_idx, end_time = None, None
        
    print("splited into {} cycle(s)".format(count))
    if count_drop > 0:
        print("dropped {} cycle(s) that has running time "\
              "less than {} minutes".format(count_drop,min_duration))
    return list_cycles

#%%

# Missing values description and data creation wrapper
from scipy.ndimage import gaussian_filter
from IPython.display import display
def create_features(dataframe, inplace=True):
    """
    Create Running_time, Delta_Time, Delta_Temperature, Temp_per_second,
    CumuSum_HeatingTemperature, CumuSum_DrumAirTemperature, 
    Inverse_HeatingTemperature, Inverse_DrumAirTemperature
    
    dataframe: Input
    Return: dataframe with added feature columns
    """
    
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()

    df.Time_stamp = pd.to_datetime(df.Time_stamp)
    df.sort_values(by='Time_stamp', inplace=True)
    
    # create Running_Time column
    df['Running_time'] = (df.Time_stamp - df.Time_stamp[0]).dt.total_seconds()
    
    # create Delta_Time column
    end_val = df.Time_stamp[-1:]
    next_val = df.Time_stamp.shift(-1).fillna(end_val)
    df['Delta_Time'] = (next_val - df.Time_stamp)
    df['Delta_Time'] = df['Delta_Time'].apply(lambda ts: ts.total_seconds())
    
    df = df[df['Delta_Time']>0] # if you can not create following columns, Just cmt that line
    # Create Drying rate
    df['DenoisedWeight_gaussian']= gaussian_filter(df.LinnenWeight,100)
    end_weight = df.DenoisedWeight_gaussian[-1:]
    next_weight = df.DenoisedWeight_gaussian.shift(-1).fillna(end_weight)
    df['Drying_rate'] = (df.DenoisedWeight_gaussian-next_weight)/df.Delta_Time
#     print("Done")
                             
    # create CumuSum_HeatingTemperature column
    area_heating = df.Delta_Time * df.HeatingTemperature
    df['CumuSum_HeatingTemperature'] = np.cumsum(area_heating)

    # create CumuSum_DrumAirTemperature
    area_druming = df.Delta_Time * df.DrumAirTemperature
    df['CumuSum_DrumAirTemperature'] = np.cumsum(area_druming)

    # create Integral_Increased_HeatingTemperature
    df['increased_temp'] = df.HeatingTemperature - df.HeatingTemperature[0]
    area_druming = df.Delta_Time * df.increased_temp
    df['Integral_Increased_HeatingTemperature'] = np.cumsum(area_druming)
    
    # create inverse
    df['Inverse_DrumAirTemperature'] = 1 / df.DrumAirTemperature
    df['Inverse_HeatingTemperature'] = 1 / df.HeatingTemperature

    # create Delta_Temp column
    end_val = df.DrumAirTemperature[-1:]
    next_val = df.DrumAirTemperature.shift(-1).fillna(end_val)
    df['Delta_Temperature'] = (next_val - df.DrumAirTemperature)

    # create Temp_per_second column
    df['Temp_per_second'] = df.Delta_Temperature/df.Delta_Time
    # create Increase temperature 
    area_increase = df.Delta_Time*(df.HeatingTemperature-df.HeatingTemperature[0])
    df['CumuSum_Increase_Temp'] = np.cumsum(area_increase)
    # create Descrease temperature
    df['Decrease_Temp'] = df.HeatingTemperature-df.DrumAirTemperature
    
    
    return df

#%%

def binning(series, from_val=0, to_val=1, bin_nums=11):
    """
    Default params: create MoistureContent_bin10 that binning
    MoistureContetnt into 11 groups: [0,0.1,0.2,etc.,1.0]
    """
    # define binning steps
    bins = np.linspace(from_val,to_val,bin_nums)
    bins = np.round(bins,1) 

    # from Series, and binning values, return bin index of each value
    bin_idx = np.digitize(series, bins,right=True)
    bin_cat = bins.astype("str")

    # get real binning values
    bin_series = list(map(lambda idx : bin_cat[idx], bin_idx))
    return bin_series


def calculate_MC(dataframe,clothes_weight=11, 
                  mc_margin = [0.2,0.06],
                  inplace=True, time_step = 10):
    """
    Create MoistureContetnt, MoistureContent_bin10, MoistureContent_class
    
    dataframe: Input
    Return: dataframe with added label columns
    """
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
        
    # create MoistureContent column
    DenoisedWeight_gaussian = gaussian_filter(df.LinnenWeight,100)
    df['MC'] = (DenoisedWeight_gaussian - clothes_weight)/clothes_weight
    df = df.loc[df.MC >=0.06]
    df = df.iloc[::time_step,:]
    # create MoistureContent_bin10
#     df["MC_bin10"] = binning(df['MC'])
    
#     # create MoistureContent_class
#     df.loc[df["MC"]>=mc_margin[0], "MC_class"] = "Wet"
#     df.loc[(df["MC"]<mc_margin[0]) & (df["MC"]>mc_margin[1]), "MC_class"] = "Damp"
#     df.loc[df["MC"]<=mc_margin[1], "MC_class"] = "Dry"
    return df

#%%

from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing

def transformer(dataframe, feature_columns, label_column, pca = False, inplace=True):
    """
    Apply data transformation (mix_max_scale + PCA) and return features, labels
    
    Params
    ------
    dataframe: train, validation or test full dataframe 
        (include features and label)
    feature_columns: list
        list of feature column names
    label_column: list
        list of label column name
    Return
    ------
    feature_PCA: pandas.DataFrame
        processed features
    label: pandas.DataFrame
    """
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
        
    # one hot encode
    cate_column = "Machine_Mode"
    encoded = pd.get_dummies(df[cate_column], prefix=cate_column)
    df = pd.concat([df, encoded],axis=1)
    
    # check and create mising column if needed:
    # not all dataframes have the same modes, especially after splited to train-test
    # a manually column creation is neccesary for ensure the homogeneity between data
    
    must_have_vals = [3,4,7]
    must_have_cols = list(map(lambda each_mode: f"{cate_column}_{each_mode}",
                              must_have_vals))
    
    default_value = 0
    for col in must_have_cols:
        if col not in list(df.columns):
            df[col] = default_value
    
    # feature selection
    features = df[feature_columns]

    # label selection
    label = df[label_column]
    
    # feature normalization
    features = features.astype("float64")
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(features)
    features = pd.DataFrame(scaled_values, columns=features.columns)
        
    # PCA
    if pca:
        pca = PCA(n_components=3)
        features = pd.DataFrame(pca.fit_transform(features))
    return features, label

#%%

def concat(dataframe_list, reset_index=True):
    concat_df = pd.concat(dataframe_list, axis = 0)
    if reset_index:
        concat_df.reset_index(drop=True, inplace=True)
    return concat_df

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
    df_hasnan = df.isna().any().any()
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

def plot_data(df, fig_height=4, no_of_col=2,verbose=True):
    """
    Plot all data values of columns in df

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
#%%%%

import numpy as np
import matplotlib.pyplot as plt
    
def plot_result(y_test, y_pred):

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

    plt.xlabel("Real value")
    plt.ylabel("Predicted value")

    plt.show()