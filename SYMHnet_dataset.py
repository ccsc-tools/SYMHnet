import pandas as pd 
import numpy as np 

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from SYMHnet_utils import  * 

class SYMHnetDataset:
    model_name= None 
    num_hours = None 
    columns_names = None
    fill_values =[9999.99,9999.99,9999.99,9999.99,9999.99,9999.99,99999.9,999.99,9999999,99.99,999.99]
    def __init__(self,model_name='SYMHnetDataset', num_hours=1, columns_names=None):
        self.model_name = model_name
        self.num_hours = num_hours 
        self.columns_names = columns_names 

    def clean_filled_values(self, data,columns_names):
        for i in range(len(columns_names)):
            c = columns_names[i]
            data = data.loc[~data[c].isin(self.fill_values)]
        return data
    
    def get_train_data(self,sample_size=None,data_filter=None):
        
        train_data = get_symh_data_from_file(self.num_hours, 'train',sample_size=sample_size,data_filter=data_filter)
        train_data = self.clean_filled_values(train_data, self.columns_names)
        train_data = train_data[columns_names].to_numpy()
        
        train_data = get_symh_data_from_file(num_hours, 'train',sample_size=100000,data_filter=[y for y in range(2010,2022)])
        train_data = clean_filled_values(train_data, columns_names)
        train_data = train_data[columns_names].to_numpy()        
        return train_data 
    
    def get_test_data(self,to_nnumpy=False):
        test_data = get_symh_data_from_file(self.num_hours, 'test',sample_size=None,data_filter=None)
        test_data = self.clean_filled_values(test_data, self.columns_names)
        test_data = test_data.reset_index()
        if to_numpy:
            test_data = test_data.to_numpy()
        return test_data 
        
                
    def split_train_validation(self, data_array: np.ndarray, train_size: float=0.9, val_size: float=0.1):
        num_time_steps = data_array.shape[0]
        num_train, num_val = (
            int(num_time_steps * train_size),
            int(num_time_steps * val_size),
        )
        train_array = data_array[:num_train]
        mean, std = train_array.mean(axis=0), train_array.std(axis=0)
    
        val_array = data_array[num_train : (num_train + num_val)]
    
        return train_array, val_array

    def create_tf_dataset(self,
        data_array: np.ndarray,
        input_sequence_length: int=1,
        forecast_horizon: int=1,
        batch_size: int = 128,
        shuffle=False,
        multi_horizon=False,
    ):
        inputs = timeseries_dataset_from_array(
            np.expand_dims(data_array[:-forecast_horizon], axis=-1),
            None,
            sequence_length=input_sequence_length,
            shuffle=False,
            batch_size=batch_size,
        )
    
        target_offset = (
            input_sequence_length
            if multi_horizon
            else input_sequence_length + forecast_horizon - 1
        )
        target_seq_length = forecast_horizon if multi_horizon else 1
        targets = timeseries_dataset_from_array(
            data_array[target_offset:],
            None,
            sequence_length=target_seq_length,
            shuffle=False,
            batch_size=batch_size,
        )
    
        dataset = tf.data.Dataset.zip((inputs, targets))
        if shuffle:
            dataset = dataset.shuffle(100)
    
        return dataset.prefetch(16).cache()
