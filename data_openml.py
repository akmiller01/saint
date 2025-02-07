import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import os


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2], forecasting=False, country_fe=True, year_fe=False):
    
    np.random.seed(seed)
    try:
        ds_id = int(ds_id)
        dataset = openml.datasets.get_dataset(ds_id)
    
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    except ValueError:
        dataset = pd.read_csv(os.path.join("data", "{}.csv".format(ds_id)))
        if not country_fe:
            dataset.pop('iso3')
        if not year_fe:
            dataset.pop('year')

        y = dataset.pop(dataset.columns[0])
        X = dataset
        small_uniques = [uniques < 20 for uniques in dataset.apply(pd.Series.nunique).to_list()]
        object_dtypes = [dtype == 'O' for dtype in X.dtypes.tolist()]
        categorical_indicator = [a or b for a, b in zip(small_uniques, object_dtypes)]
        # if ds_id == "tripartite_bigram":
        #     categorical_indicator = [False, False, False, False, True, False, False, False]
        categorical_columns = X.columns[categorical_indicator].tolist()
        for i in range(0, len(categorical_indicator)):
            print(X.columns[i], ": Categorical" if categorical_indicator[i] else ": Numerical")

        if forecasting:
            forecast_dataset = pd.read_csv(os.path.join("data", "{}_forecasting.csv".format(ds_id)))
            if not country_fe:
                forecast_dataset.pop('iso3')
            if not year_fe:
                forecast_dataset.pop('year')
            y_forecast = forecast_dataset.pop(forecast_dataset.columns[0])
            forecast_categories = forecast_dataset.pop(forecast_dataset.columns[0])
            X_forecast = forecast_dataset

        for categorical_column in categorical_columns:
            X[categorical_column] = X[categorical_column].astype('str')
            if forecasting:
                X_forecast[categorical_column] = X_forecast[categorical_column].astype('str')

    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")
        if forecasting:
            X_forecast[col] = X_forecast[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    if forecasting:
        forecast_temp = X_forecast.fillna("MissingValue")
        forecast_nan_mask = forecast_temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
        if forecasting:
            X_forecast[col] = X_forecast[col].fillna("MissingValue")
            X_forecast[col] = l_enc.fit_transform(X_forecast[col].values)
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
        if forecasting:
            X_forecast.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if forecasting:
        y_forecast = y_forecast.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
        if forecasting:
            y_forecast = l_enc.fit_transform(y_forecast)
    if task == 'regression':
        y_mean, y_std = y.mean(0), y.std(0)
        y = (y - y_mean) / y_std
        if forecasting:
            y_forecast = (y_forecast - y_mean) / y_std
    else:
        y_mean, y_std = (None, None)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)
    if forecasting:
        X_forecast, y_forecast = data_split(X_forecast, y_forecast,forecast_nan_mask,X_forecast.index)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    if forecasting:
        return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, X_forecast, y_forecast, train_mean, train_std, y_mean, y_std
    else:
        return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, y_mean, y_std




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

