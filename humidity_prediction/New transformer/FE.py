def create_features(dataframe,feature_columns=None):
    """
    Create more columns/features. Some might be redundant but still be kept
    for now for data exploitation
    dataframe: Input
    
    Return: dataframe with added feature columns
    """
    
    df = dataframe.copy()

    df.Time_stamp = pd.to_datetime(df.Time_stamp)
    df.sort_values(by='Time_stamp', inplace=True)
    
    # create Running_Time column
    df['Running_Time'] = (df.Time_stamp - df.Time_stamp[0]).dt.total_seconds()
    
    # Delta_Time
    end_val = df.Time_stamp[-1:]
    next_vals = df.Time_stamp.shift(-1).fillna(end_val)
    df['Delta_Time'] = (next_vals - df.Time_stamp)
    df['Delta_Time'] = df['Delta_Time'].apply(lambda ts: ts.total_seconds())
    
#     df = df[df['Delta_Time']>0] 
    
    # Integral_HeatingTemperature column
    rectangle_area = df.Delta_Time * df.HeatingTemperature
    df['Integral_HeatingTemperature'] = np.cumsum(rectangle_area)

    # Integral_DrumAirTemperature
    rectangle_area = df.Delta_Time * df.DrumAirTemperature
    df['Integral_DrumAirTemperature'] = np.cumsum(rectangle_area)
    
    # Integral_Increased_HeatingTemperature
    Increased_HeatingTemperature = df.HeatingTemperature - df.HeatingTemperature[0]
    rectangle_area = df.Delta_Time * Increased_HeatingTemperature
    df['Integral_Increased_HeatingTemperature'] = np.cumsum(rectangle_area)
    
    # Integral_Increased_DrumAirTemperature
    Increased_DrumAirTemperature = df.DrumAirTemperature - df.DrumAirTemperature[0]
    rectangle_area = df.Delta_Time * Increased_DrumAirTemperature
    df['Integral_Increased_DrumAirTemperature'] = np.cumsum(rectangle_area)

    # Integral_LossTemperature
    LossTemperature = df.HeatingTemperature - df.DrumAirTemperature
    rectangle_area = df.Delta_Time * LossTemperature
    df['Integral_LossTemperature'] = np.cumsum(rectangle_area)
    
    # TEST
    df['Increased_DrumAirTemperature'] = df.DrumAirTemperature - df.DrumAirTemperature[0]
    
    
    # Inverse_DrumAirTemperature
    df['Inverse_DrumAirTemperature'] = 1 / df.DrumAirTemperature
    # Inverse_HeatingTemperature
    df['Inverse_HeatingTemperature'] = 1 / df.HeatingTemperature
    
    non_zero_delta_time = df.Delta_Time.copy()
    non_zero_delta_time.loc[df.Delta_Time==0] = 1
    
    # Delta_HeatingTemperature
    end_val = df.HeatingTemperature[-1:]
    next_vals = df.HeatingTemperature.shift(-1).fillna(end_val)
    df['Delta_HeatingTemperature'] = (next_vals - df.HeatingTemperature)
    # Deriv_HeatingTemperature
    df['Deriv_HeatingTemperature'] = df.Delta_HeatingTemperature / non_zero_delta_time
    # Integral_Deriv_HeatingTemperature
    rectangle_area = df.Delta_Time * df.Deriv_HeatingTemperature
    df['Integral_Deriv_HeatingTemperature'] = np.cumsum(rectangle_area)
    
    
    # Delta_DrumAirTemperature
    end_val = df.DrumAirTemperature[-1:]
    next_vals = df.DrumAirTemperature.shift(-1).fillna(end_val)
    df['Delta_DrumAirTemperature'] = (next_vals - df.DrumAirTemperature)
    # Deriv_DrumAirTemperature
    df['Deriv_DrumAirTemperature'] = df.Delta_DrumAirTemperature / non_zero_delta_time
    # Integral_Deriv_DrumAirTemperature
    rectangle_area = df.Delta_Time * df.Deriv_DrumAirTemperature
    df['Integral_Deriv_DrumAirTemperature'] = np.cumsum(rectangle_area)
    
    return df

#%%

from scipy.ndimage import gaussian_filter
import numpy as np

def binning(series, from_val=0, to_val=10, bin_nums=101):
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
                  mc_margin = [0.2,0.06]):
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
    df = dataframe.copy()
        
    # create MoistureContent column
    DenoisedWeight_gaussian = gaussian_filter(df.LinnenWeight,120)
    df['MC'] = (DenoisedWeight_gaussian - dry_clothes_weight)/dry_clothes_weight

    # create MoistureContent_bin10
    df["MC_bin10"] = binning(df['MC'])
    
    # create MoistureContent_class
    df.loc[df["MC"]>=mc_margin[0], "MC_class"] = "Wet"
    df.loc[(df["MC"]<mc_margin[0]) & (df["MC"]>mc_margin[1]), "MC_class"] = "Damp"
    df.loc[df["MC"]<=mc_margin[1], "MC_class"] = "Dry"
    
    return df

#%%

from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class Transformer:
    
    def __init__(self,apply_PCA=False,normalize=True):
        """
        apply_PCA: boolean
            apply PCA or not
        normalize: boolean
            apply normalization or not
        """
        self.apply_PCA = apply_PCA
        self.normalize = normalize
        if self.normalize:
            self.scaler = MinMaxScaler() # intitial MinmaxScaler
        if self.apply_PCA:
            # PCA and keep n_components that retain more than 95% information
            self.pca = PCA(n_components=0.95,svd_solver='full') 

    
    def _fit(self, data):
        x = data.copy()
        if self.normalize:
            x = self.scaler.fit_transform(x)
        if self.apply_PCA:
            x = self.pca.fit_transform(x) # X is data, call fit function of PCA
    
    
    def _transform(self,data):
        x = data.copy()
        if self.normalize:
            x = self.scaler.transform(x)
            x = pd.DataFrame(x, columns=data.columns)
        if self.apply_PCA:
            x = self.pca.transform(x) # transform PCA data
        return x
    
    @staticmethod
    def create_features_and_labels(data_list,dry_clothes_weight):
        """
        create columns
        
        params
        ------
        data_list: list of pandas.DataFrame
            cycles in a list
        dry_clothes_weight: integer or list of integer
            initial dry weight of cycles. 
            Single integer => same dry_clothes_weight of cycles in data_list
            List of integer =>  respectively values to cycles in data_list
        """
        if not isinstance(data_list,list):
            data_list = [data_list]
            
        if isinstance(dry_clothes_weight,list):
            for i in range(len(data_list)):
                data_list[i] = create_features(data_list[i])
                data_list[i] = create_labels(data_list[i], dry_clothes_weight[i])
                
        else:
            for i in range(len(data_list)):
                data_list[i] = create_features(data_list[i])
                data_list[i] = create_labels(data_list[i], dry_clothes_weight)
                
        return data_list
       
        
    @staticmethod
    def concat(data_list, reset_index=True):
        """
        concatenate dataframes in "data_list" to a dataframe
        
        params
        ------
        data_list: list of pandas.DataFrame
            pass
        reset_index: boolean
            reset the index of concatenated dataframe
            
        return
        ------
        concat_df: pandas.DataFrame
            concatenated dataframe
        """
        concat_df = pd.concat(data_list, axis = 0)
        if reset_index:
            concat_df.reset_index(drop=True, inplace=True)
        return concat_df
    
    @staticmethod
    def one_hot_encode(data):
        """one hot encode Machine_Mode"""
        cate_column = "Machine_Mode"
        encoded = pd.get_dummies(data[cate_column], prefix=cate_column)
        data_raw = pd.concat([data, encoded],axis=1)

        # check and create mising column if needed:
        # not all dataframes have the same modes, especially after splited to train-test
        # a manually column creation is neccesary for ensure the homogeneity between data

        must_have_vals = [3,4,7]
        must_have_cols = list(map(lambda each_mode: f"{cate_column}_{each_mode}",
                                  must_have_vals))

        default_value = 0
        for col in must_have_cols:
            if col not in list(data.columns):
                data[col] = default_value # fill all missing columns with 0
        
        return data
    
    @staticmethod
    def select_features_and_labels(data, feature_columns, label_column):
        """
        data: dataframe
        feature_columns: list of string name of features
        label_column: list of string name of a label
        """
        # feature selection
        features = data[feature_columns]
        features = features.astype("float64")
        # label selection
        labels = data[label_column]
        return features, labels
    
    @staticmethod
    def preprocess(data_list, dry_clothes_weight,
                      feature_columns, label_column):
        """
        data_list: list of pandas.DataFrame
        dry_clothes_weight: kg of initial dry clothes weights
        feature_columns: list of string name of features
        label_column: list of string name of a label
        
        return: tuple of preprocessed features and labels
        """
        data_list = Transformer.create_features_and_labels(data_list, dry_clothes_weight)
        data = Transformer.concat(data_list)
        data = Transformer.one_hot_encode(data)
        features, labels = Transformer.select_features_and_labels(data,
                                                                 feature_columns,
                                                                 label_column)
        return features, labels
    
    
    def fit(self, data_list, dry_clothes_weight,
            feature_columns, label_column):
        """
        data_list: list of pandas.DataFrame
        dry_clothes_weight: kg of initial dry clothes weights
        feature_columns: list of string name of features
        label_column: list of string name of a label
        """
        features,labels = Transformer.preprocess(data_list, dry_clothes_weight,
                                         feature_columns, label_column)
        self._fit(features)
    
    def transform(self, data_list, dry_clothes_weight,
                  feature_columns, label_column):
        """
        data_list: list of pandas.DataFrame
        dry_clothes_weight: kg of initial dry clothes weights
        feature_columns: list of string name of features
        label_column: list of string name of a label
        
        return tuple of features and labels
        """
        
        features,labels = Transformer.preprocess(data_list, dry_clothes_weight,
                                         feature_columns, label_column)
        features = self._transform(features)
        return features, labels
    
    def fit_transform(self, data_list, dry_clothes_weight,
                  feature_columns, label_column):
        """
        fit and transform, reduce the time to apply preprocessing
        
        data_list: list of pandas.DataFrame
        dry_clothes_weight: kg of initial dry clothes weights
        feature_columns: list of string name of features
        label_column: list of string name of a label
        
        return tuple of features and labels
        """
        
        features,labels = Transformer.preprocess(data_list, dry_clothes_weight,
                                         feature_columns, label_column)
        self._fit(features)
        features = self._transform(features)
        return features, labels