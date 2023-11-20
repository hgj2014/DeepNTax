#!/usr/bin/env python
# coding: utf-8


import csv
import gc
import os
import glob
import itertools
import pickle
import random
import operator
import copy
import platform
from collections import OrderedDict
import sys
import json
import math
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import time
import tensorflow as tf

# def configure_gpus():
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             print(e)

# configure_gpus()

# gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(777)
warnings.filterwarnings("ignore")

from deepntax import configuration
from deepntax import loss_and_metric
from deepntax import build_phylo_networks

import keras.backend as k  
from pkg_resources import resource_filename


logging.basicConfig(format = '[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s',
                    level=logging.DEBUG)
log = logging.getLogger()


# # Functions
# ## Data split
def split_md_test_set(total_set, target_str, y_colname, n_fold, randomSeed):
    """
    Splits the medical information dataset into a training set and a test set.

    Args:
        medicalInfo (DataFrame): The medical information dataset.
        target (str): The target variable.
        y_col (str): The column name of the target variable.
        n_fold (int): The number of folds for cross-validation.
        seed (int): The seed for the random number generator.

    Returns:
        Tuple[DataFrame, DataFrame]: The training set and the test set.
    """

    print(stratify_cols, covariates)
    
    total_set.index = total_set[sample_col].tolist()
    
    total_set_subset = total_set.loc[total_set[y_colname].isin([int(x) for x in target_str.split('_')]), :]
    
    # subset medical info only with covariates without NA
    total_set_subset = total_set_subset.loc[pd.notna(total_set_subset.loc[:, covariates]).all(axis = 1), :]
    
    md_set, test_set = train_test_split(total_set_subset, test_size=1/3, random_state=randomSeed, stratify=total_set_subset[stratify_cols])
    
    md_set_copy = md_set.copy()

    count_summary = pd.DataFrame(md_set_copy.groupby(stratify_cols)[y_colname].count())
    
    if any(count_summary[y_colname] < n_fold):
        not_enough_cols = list(count_summary[y_colname].loc[count_summary[y_colname] < n_fold].index.values[0])
        not_enough_data = md_set_copy.loc[(md_set_copy[stratify_cols] == not_enough_cols).sum(axis=1) == len(stratify_cols), ]
        enough_data = md_set_copy.loc[~md_set_copy[sample_col].isin(not_enough_data[sample_col]), ]
    else:
        enough_data = md_set_copy
        not_enough_data = []

    set_split_ids = []
    
    cv = list(np.arange(1, n_fold+1))
    
    for each_cv in reversed(cv[1:]) :
        enough_data, temp_train = train_test_split(enough_data, test_size=int(len(enough_data)/each_cv), random_state=randomSeed, stratify=enough_data[stratify_cols])
        set_split_ids.append(temp_train[sample_col].tolist())
    set_split_ids.append(enough_data[sample_col].tolist())

    for each_cv in range(len(not_enough_data)):
        set_split_ids[each_cv].append(not_enough_data.iloc[each_cv, ][sample_col])

    print('current index:', randomSeed)
    print('shape of md set:', md_set.shape, 'shape of test set:', test_set.shape)
    print('md set split count:', [len(x) for x in set_split_ids], '\n')
    
    md_set.insert(len(md_set.columns), 'set', '', True)
    
    
    for each_cv in range(len(set_split_ids)):
        each_set = set_split_ids[each_cv]
        md_set.loc[md_set[sample_col].isin(each_set), 'set'] = each_cv

    return md_set, test_set


def split_train_val_set(md_set, val_idx):
    # split train and validation set
    train_set = md_set.loc[md_set['set'] != val_idx, :]
    val_set = md_set.loc[md_set['set'] == val_idx, :]
    
    return train_set, val_set


# ## Data binning
def data_binning(medInfo, otu_data, target_str, hac_index):
    """
    Bins the data in the medical information and OTU relative abundance tables.

    Args:
        medicalInfo (DataFrame): The medical information dataset.
        otuRA_table (DataFrame): The OTU relative abundance table.
        target (str): The target variable.
        otus (list): The list of OTUs.

    Returns:
        DataFrame: The binned data.
    """

    target1, target2 = target_str.split('_')
    
    # scaling y value to {0, 1}
    if int(target1) == 0 and int(target2) == 2: 
        correction = 2
    elif int(target1) == 1:
        correction = 1
    else:
        correction = 0
    
    sample_list = medInfo[sample_col].tolist()
    otu_subset = otu_data.loc[medInfo[sample_col], ]
    X_df = pd.concat([medInfo[covariates], otu_subset.loc[:, hac_index]], axis=1).astype(np.float32)
    y_list = medInfo[y_col].tolist()
    if correction != 0:
        y_list = y_scaling(y_list, correction)
                
    y_arr = np.array(y_list, dtype=int)
    
    return [sample_list, X_df, y_arr]


def y_scaling(init_y, correction):
    # scaling y value to {0, 1}
    # input: initial y value, correction value
    # output: scaled y value

    if correction == 1:
        return [int(each_y - correction) for each_y in init_y]
    elif correction == 2:
        return [int(each_y / correction) for each_y in init_y]



def WriteCSV(output_file, line):
    # csv write function
    # input: file name with extension and what to write
    csv_out_file = open(output_file, 'a', newline = '')
    filewriter = csv.writer(csv_out_file)
    filewriter.writerow(line)
    csv_out_file.close()



def format_pred_file(pred_value, sample_id, actual_value):
    # format prediction file
    # input: prediction value, sample id, actual value
    # output: dataframe with sample id, prediction value, actual value

    pred_file = pd.DataFrame(pred_value)
    pred_file['Sample_id'] = sample_id
    pred_file['actual'] = actual_value
    
    return pred_file



def train_DeepNTax(train_data, val_data, test_data, model_combination, md_data, tree_info, is_test, is_cov):
    """
    Trains the DeepNTax model.

    Args:
        train_set (DataFrame): The training set.
        val_set (DataFrame): The validation set.
        test_set (DataFrame): The test set.
        hyperparmeter_set (dict): The set of hyperparameters for the model.
        md_set (DataFrame): The metadata set.
        tree_info (dict): The tree information for the model.
        is_test (bool, optional): Whether or not this is a test run. Defaults to True.
        is_cov (bool, optional): Whether or not there are covariates. Defaults to False.

    Returns:
        Tuple[Model, History]: The trained model and the history of the training process.
    """

    now = datetime.now()
    print("Model compile start time : ", now.strftime("%y%m%d-%H%M%S"))
    
    # data loading
    train_x = train_data[1]
    train_y = train_data[2]
    train_otus = tf.reshape(train_x.loc[:, otus], [-1, train_x.loc[:, otus].shape[1]])
    train_cov = train_x.loc[:, covariates]
    
    val_x = val_data[1]
    val_y = val_data[2]
    val_otus = tf.reshape(val_x.loc[:, otus], [-1, val_x.loc[:, otus].shape[1]])
    val_cov = val_x.loc[:, covariates]
    
    md_x = md_data[1]
    md_y = md_data[2]
    md_otus = tf.reshape(md_x.loc[:, otus], [-1, md_x.loc[:, otus].shape[1]])
    md_cov = md_x.loc[:, covariates]
    
    test_x = test_data[1]
    test_y = test_data[2]
    test_otus = tf.reshape(test_x.loc[:, otus], [-1, test_x.loc[:, otus].shape[1], 1])
    test_cov = test_x.loc[:, covariates]
    
    # if there is covariates, use it
    if is_cov:
        train_input = [train_otus, train_cov]
        val_input = [val_otus, val_cov]
        md_input = [md_otus, md_cov]
        test_input = [test_otus, test_cov]
    else:
        train_input = [train_otus]
        val_input = [val_otus]
        md_input = [md_otus]
        test_input = [test_otus]
        
    print(len(test_input))

    network_type = getattr(build_phylo_networks, network_info_dict['model_info']['network_class'].strip())
    
    model = network_type(network_info_dict, tree_info, log, num_classes=1,
                         tree_level_list = tree_info.columns.to_list(),
                         is_covariates=is_cov, covariate_names = np.array(covariates), verbose=True, with_level=bool(network_info_dict['model_info']['with_level']))

    model.model_compile()
    
    model_history = model.model.fit(train_input, train_y, batch_size=int(model_combination['batch_size']), verbose=0, 
                                    epochs=int(model_combination['pred_max_epoch']), validation_data = (val_input, val_y), workers=2, use_multiprocessing=True,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(model_combination['pred_patience']), 
                                                                                restore_best_weights=True)])

    # model prediction and recording results
    train_pred = np.array(model.model.predict(train_input, verbose=0, batch_size=len(train_otus)), np.float64)
    val_pred = np.array(model.model.predict(val_input, verbose=0, batch_size=len(val_otus)), np.float64)
    md_pred = np.array(model.model.predict(md_input, verbose=0, batch_size=len(md_otus)), np.float64)
    test_pred = np.array(model.model.predict(test_input, verbose=0, batch_size=len(test_otus)), np.float64)
    
    if is_test:
        train_pred_df = format_pred_file(train_pred, train_data[0], train_data[2])
        train_pred_df.to_csv(pred_save_dir + 'train_pred.csv', index = False)

        val_pred_df = format_pred_file(val_pred, val_data[0], val_data[2])
        val_pred_df.to_csv(pred_save_dir + 'val_pred.csv', index = False)

        test_pred_df = format_pred_file(test_pred, test_data[0], test_data[2])
        test_pred_df.to_csv(pred_save_dir + 'test_pred.csv', index = False)
        
        del train_pred_df, val_pred_df, test_pred_df

    result_dict["feature"] = train_x.shape[1]
    result_dict['pred_epoch'] = len(model_history.history['loss']) - int(model_combination['pred_patience'])

    _, result_dict["train_acc"], result_dict["train_sensitivity"], result_dict["train_specificity"] = tuple(model.model.evaluate(
        train_input, train_y, batch_size=len(train_otus), verbose=0))
    _, result_dict["val_acc"], result_dict["val_sensitivity"], result_dict["val_specificity"] = tuple(model.model.evaluate(
        val_input, val_y, batch_size=len(val_otus), verbose=0))
    _, result_dict["md_acc"], result_dict["md_sensitivity"], result_dict["md_specificity"] = tuple(model.model.evaluate(
        md_input, md_y, batch_size=len(md_otus), verbose=0))
    _, result_dict["test_acc"], result_dict["test_sensitivity"], result_dict["test_specificity"] = tuple(model.model.evaluate(
        test_input, test_y, batch_size=len(test_otus), verbose=0))
    
    result_dict["train_AUC"] = roc_auc_score(train_y, train_pred.ravel())
    result_dict["val_AUC"] = roc_auc_score(val_y, val_pred.ravel())
    result_dict["md_AUC"] = roc_auc_score(md_y, md_pred.ravel())
    result_dict["test_AUC"] = roc_auc_score(test_y, test_pred.ravel())
    
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    return model, model_history


now = datetime.now()
print("Code start time : ", now.strftime("%Y-%m-%d %H:%M:%S"))

network_config_file = sys.argv[1]
path_config_file = sys.argv[2]


print(network_config_file, path_config_file)

# load configuration file
network_config = configuration.Configurator(network_config_file, log)
path_config = configuration.Configurator(path_config_file, log)

network_config.set_config_map(network_config.get_section_map())
path_config.set_config_map(path_config.get_section_map())

network_info_dict = network_config.get_config_map()
path_info_dict = path_config.get_config_map()

data_dir = path_info_dict['data_info']['data_path']

# load data
otuRA_table = pd.read_csv(data_dir + path_info_dict['data_info']['x_path'])
medicalInfo = pd.read_csv(data_dir + path_info_dict['data_info']['y_path'])
tree_info = pd.read_csv(data_dir + path_info_dict['data_info']['tree_info_path'])

# set variables
stratify_cols = [x.strip() for x in network_info_dict['model_info']['stratify_cols'].split(',')] # ["Sample_group_num", "Sample_source", "Sex"]
y_col = 'Sample_group_num'
sample_col = 'Sample_id'
otus = otuRA_table.columns.tolist()

n_fold = 5

result_dict = dict()
val_auc_baseline = 0


# Data transformation: minmax(arcsin(sqrt(x)))
otuRA_table = np.sqrt(otuRA_table)
otuRA_table = np.arcsin(otuRA_table)
otuRA_table = (otuRA_table - otuRA_table.min(axis=0)) / (otuRA_table.max(axis=0) - otuRA_table.min(axis=0))


# Directories to save the results of model
result_dir = path_info_dict['save_info']['model_dir'] + 'example/'
if os.path.exists(result_dir) != True:
    os.makedirs(result_dir)

model_save_dir = result_dir + 'models/'
if os.path.exists(model_save_dir) != True:
    os.makedirs(model_save_dir)

pred_save_dir = result_dir + 'pred/'
if os.path.exists(pred_save_dir) != True:
    os.makedirs(pred_save_dir)


main_col = list(network_info_dict['hyperparmeter_set'].keys())

main_col = main_col + ['pred_epoch', 'train_acc', 'train_sensitivity', 'train_specificity', 'train_AUC',
                       'val_acc', 'val_sensitivity', 'val_specificity', 'val_AUC',
                       'md_acc', 'md_sensitivity', 'md_specificity', 'md_AUC',
                       'test_acc', 'test_sensitivity', 'test_specificity', 'test_AUC']


output_file = result_dir + "model_result.csv"
WriteCSV(output_file, main_col)

with tf.device('/device:GPU:0'): # set gpu to use

    target = network_info_dict['model_info']['target'] 

    # save model hyperparameter
    for each_key, each_value in zip(network_info_dict['hyperparmeter_set'].keys(), network_info_dict['hyperparmeter_set'].values()):
        network_info_dict['architecture_info'][each_key] = each_value

    # set covariates
    covariates = ['Age', 'Sample_sex_binary']

    if len(covariates) == 0:
        is_cov = False
    else:
        is_cov = True
        
    # Dataset split

    # 2:1
    md_medicalInfo, test_medicalInfo = split_md_test_set(medicalInfo, target, y_col, n_fold, 777)

    md_set = data_binning(md_medicalInfo, otuRA_table, target, otus)
    test_set = data_binning(test_medicalInfo, otuRA_table, target, otus)

    val_index = 0

    print('\nvalidation index: ', val_index)

    # 4:1
    train_medicalInfo, val_medicalInfo = split_train_val_set(md_medicalInfo, val_index)

    train_set = data_binning(train_medicalInfo, otuRA_table, target, otus)
    val_set = data_binning(val_medicalInfo, otuRA_table, target, otus)

    print('\n\ntest set')

    model, model_history = train_DeepNTax(train_set, val_set, test_set, network_info_dict['hyperparmeter_set'], md_set, tree_info, is_test=True, is_cov=is_cov)

    model.model.save_weights(model_save_dir + 'model_weights.h5')

    # format into result and write csv
    dup_cols = [x for x in network_info_dict['hyperparmeter_set'].keys() if x in main_col]

    for each_col in dup_cols:
        result_dict[each_col] = network_info_dict['hyperparmeter_set'][each_col]

    # write performance result
    sorted_result_dict = OrderedDict((k, result_dict[k]) for k in main_col)
    result_list = list(sorted_result_dict.values())
    WriteCSV(output_file, result_list)

    del train_set, val_set, md_set, test_set, model_history

    tf.keras.backend.clear_session()
    gc.collect()
        