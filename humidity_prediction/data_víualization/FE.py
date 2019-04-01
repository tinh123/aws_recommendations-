import pandas as pd
import numpy as np
from datetime import datetime as dt

def cycle_spliter(dataframe, sort = False, to_files=False, padding = False,
                  min_duration=0, running_mode="pi"):
    """
    Params
    ------
    dataframe: pandas.DataFrame
        Input data, must have "Machine_Mode", "Time_stamp" columns
    sort: boolean
        sort dataframe by Time_stamp
    to_files: boolean
        save cycle-based splited dataframe into csv files, name is cycle_[order]
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
    
    
    df['Machine_Mode'] = df.Machine_Mode.astype(str)
    df['Machine_Mode'] = df['Machine_Mode'].apply(lambda row: row.strip())
    
    # Drop NaN rows in Machine_mode
    if df.Machine_Mode.hasnans:
        df.drop(index=df.loc[df.Machine_Mode.isna()].index,axis=0, inplace=True)

    if "Cycle_Start_Time" not in df.columns:
        df.insert(2,"Cycle_Start_Time",np.nan)
    if "Cycle_End_Time" not in df.columns:
        df.insert(3,"Cycle_End_Time",np.nan)
        
    df.reset_index(drop=True,inplace=True)

    start_idx, start_time = None, None
    end_idx, end_time = None, None
    count, count_drop = 0, 0
    list_cycles=[]
    for this_index, this_mode in enumerate(df.Machine_Mode):
        try:
            next_index = this_index + 1
            next_mode = df.Machine_Mode[next_index]
        except KeyError: # end of df, stop loop
            break
            
        # define start time
        start_default = this_mode == "0" and next_mode == "3"
        start_pi = this_mode == "Stop" and next_mode == "Start"
        if start_default or start_pi:
            if padding:
                start_idx = this_index # val = 0
            else:
                start_idx = next_index # val = 3
            start_time = df.Time_stamp[start_idx]
            
        # define end time
        stop_default = this_mode != "5" and next_mode == "5" and start_idx != None
        stop_pi = this_mode == "Start" and next_mode == "Stop" and start_idx != None
        if stop_default or stop_pi:
            if padding:
                end_idx = next_index # val == 5
            else:
                end_idx = this_index # val != 5
            end_time = df.Time_stamp[end_idx]
            
            # only cycles last >= min_duration are kept            
            duration = round((end_time - start_time).total_seconds()/60,1)
            if duration >= min_duration:
                df.loc[start_idx:end_idx, ['Cycle_Start_Time','Cycle_End_Time']] = [start_time, end_time]
                cycle = df.loc[start_idx:end_idx]
                cycle.reset_index(drop=True,inplace=True)
                list_cycles.append(cycle)
                count += 1
                print("start time: {} \t end time: {} \t duration: {} mins".format(
                    start_time,end_time, duration))

                if to_files:
                    exec("df[start_idx:end_idx+1].to_csv('cycle_{count}.csv', index=False)".format(count=count))
            
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

# Missing-values description and data creation wrapper

import pandas as pd

def create_features(dataframe, inplace=True):
    """
    Create more columns/features. Some might be redundant but still be kept
    for now for data exploitation
    
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
    
    # try to comment out the next line in case of any memory error
    df = df[df['Delta_Time']>0] 
    
    # create Integral_HeatingTemperature column
    rectangle_area = df.Delta_Time * df.HeatingTemperature
    df['Integral_HeatingTemperature'] = np.cumsum(rectangle_area)

    # create Integral_DrumAirTemperature
    rectangle_area = df.Delta_Time * df.DrumAirTemperature
    df['Integral_DrumAirTemperature'] = np.cumsum(rectangle_area)
    
    # create Integral_Increased_HeatingTemperature
    Increased_HeatingTemperature = df.HeatingTemperature - df.HeatingTemperature[0]
    rectangle_area = df.Delta_Time * Increased_HeatingTemperature
    df['Integral_Increased_HeatingTemperature'] = np.cumsum(rectangle_area)
    
    # create Integral_Increased_DrumAirTemperature
#     Increased_DrumAirTemperature = 
#     rectangle_area = df.Delta_Time * Increased_DrumAirTemperature
#     df['Integral_Increased_DrumAirTemperature'] = np.cumsum(rectangle_area)
    df['Increased_DrumAirTemperature']= df.DrumAirTemperature - df.DrumAirTemperature[0]

    # create Integral_LossTemperature
    LossTemperature = df.HeatingTemperature - df.DrumAirTemperature
    rectangle_area = df.Delta_Time * LossTemperature
    df['Integral_LossTemperature'] = np.cumsum(rectangle_area)

    df['Test'] = (df.Integral_Increased_HeatingTemperature + df.Integral_LossTemperature)/2
    
    # create inverse
    df['Inverse_DrumAirTemperature'] = 1 / df.DrumAirTemperature
    df['Inverse_HeatingTemperature'] = 1 / df.HeatingTemperature

    # create Delta_Temp column
    end_val = df.DrumAirTemperature[-1:]
    next_val = df.DrumAirTemperature.shift(-1).fillna(end_val)
    df['Delta_Temperature'] = (next_val - df.DrumAirTemperature)

    # create Temp_per_second column
    df['Temp_per_second'] = df.Delta_Temperature/df.Delta_Time
    
    return df

#%%

from scipy.ndimage import gaussian_filter
import numpy as np

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


def create_labels(dataframe,dry_clothes_weight, 
                  mc_margin = [0.2,0.06],
                  inplace=True):
    """
    Create MoistureContetnt, MoistureContent_bin10, MoistureContent_class
    
    dataframe: Input
    dry_clothes_weight: int
        clothes weight without moisture
    mc_margin: list of 2 integers
        the 1st value is margin of "Wet" and "Damp"
        the 2nd value is margin of "Damp" and "Dry"
        
    Return: dataframe with added label columns
    """
    if inplace:
        df = dataframe
    else:
        df = dataframe.copy()
        
    # create MoistureContent column
    DenoisedWeight_gaussian = gaussian_filter(df.LinnenWeight,90)
    df['MC'] = (DenoisedWeight_gaussian - dry_clothes_weight)/dry_clothes_weight

    # create MoistureContent_bin10
    df["MC_bin10"] = binning(df['MC'])
    
#     # create MoistureContent_class
#     df.loc[df["MC"]>=mc_margin[0], "MC_class"] = "Wet"
#     df.loc[(df["MC"]<mc_margin[0]) & (df["MC"]>mc_margin[1]), "MC_class"] = "Damp"
#     df.loc[df["MC"]<=mc_margin[1], "MC_class"] = "Dry"
    return df

#%%

# from sklearn.decomposition import PCA
# import pandas as pd
# from sklearn import preprocessing

# def transformer(dataframe, feature_columns, label_column, pca = False, inplace=True):
#     """
#     Apply data transformation (mix_max_scale + PCA) and return features, labels
    
#     Params
#     ------
#     dataframe: train, validation or test full dataframe 
#         (include features and label)
#     feature_columns: list
#         list of feature column names
#     label_column: list
#         list of label column name
#     Return
#     ------
#     feature_PCA: pandas.DataFrame
#         processed features
#     label: pandas.DataFrame
#     """
#     if inplace:
#         df = dataframe
#     else:
#         df = dataframe.copy()
        
#     # one hot encode
#     cate_column = "Machine_Mode"
#     encoded = pd.get_dummies(df[cate_column], prefix=cate_column)
#     df = pd.concat([df, encoded],axis=1)
    
#     # check and create mising column if needed:
#     # not all dataframes have the same modes, especially after splited to train-test
#     # a manually column creation is neccesary for ensure the homogeneity between data
    
#     must_have_vals = [3,4,7]
#     must_have_cols = list(map(lambda each_mode: f"{cate_column}_{each_mode}",
#                               must_have_vals))
    
#     default_value = 0
#     for col in must_have_cols:
#         if col not in list(df.columns):
#             df[col] = default_value
    
#     # feature selection
#     features = df[feature_columns]

#     # label selection
#     label = df[label_column]
    
#     # feature normalization
#     features = features.astype("float64")
#     min_max_scaler = preprocessing.MinMaxScaler()
#     scaled_values = min_max_scaler.fit_transform(features)
#     features = pd.DataFrame(scaled_values, columns=features.columns)
        
#     # PCA
#     if pca:
#         pca = PCA(n_components=3)
#         features = pd.DataFrame(pca.fit_transform(features))
#     return features, label

# #%%

def concat(dataframe_list, reset_index=True):
    concat_df = pd.concat(dataframe_list, axis = 0)
    if reset_index:
        concat_df.reset_index(drop=True, inplace=True)
    return concat_df