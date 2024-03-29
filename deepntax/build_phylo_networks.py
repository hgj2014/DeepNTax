######################################################################
## DeepNTax
## - Build DeepNTax model network
##
## Reference
## - Keras (https://github.com/keras-team/keras)
## - DeepBiome (https://github.com/Young-won/deepbiome)
######################################################################

import time
import json
import sys
import abc
import copy
import numpy as np
import pandas as pd

import keras
import keras.callbacks

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Flatten, Lambda, Reshape, LeakyReLU
from keras.layers import Concatenate
from keras.layers import BatchNormalization, Dropout
from keras.initializers import VarianceScaling

from . import loss_and_metric
# from .utils import TensorBoardWrapper

pd.set_option('display.float_format', lambda x: '%.03f' % x)
np.set_printoptions(formatter={'float_kind':lambda x: '%.03f' % x})
     
#####################################################################################################################
# Base Network
#####################################################################################################################
class Base_Network(abc.ABC):
    """Inherit from this class when implementing new networks."""
    def __init__(self, network_info, log):
        # Build Network
        self.network_info = network_info
        self.log = log
        self.best_model_save = False
        # self.TB = TensorBoardWrapper
        self.model = None
    
    @abc.abstractmethod
    def build_model(self, verbose=True):
        # define self.model
        pass
        
    def model_compile(self):
        self.model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['loss']),
                           optimizer=getattr(tf.keras.optimizers, 
                                             self.network_info['architecture_info']['optimizer'])(lr=float(self.network_info['architecture_info']['lr']),
                                                                                           decay=float(self.network_info['model_info']['decay'])),
                           metrics=[getattr(loss_and_metric, metric.strip()) for metric in self.network_info['model_info']['metrics'].split(',')])

        self.log.info('Build Network')
        self.log.info('Optimizer = {}'.format(self.network_info['architecture_info']['optimizer']))
        self.log.info('Loss = {}'.format(self.network_info['model_info']['loss']))
        self.log.info('Metrics = {}'.format(self.network_info['model_info']['metrics']))
        
    def load_model(self, model_yaml_path, verbose=0):
        # load model from YAML
        yaml_file = open(model_yaml_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_yaml)
        # self.model = model_from_yaml(loaded_model_yaml)
        if verbose:
            self.log.info(self.model.summary())
    
    def save_model(self, model_yaml_path):
        model_yaml = self.model.to_json()
        with open(model_yaml_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
    
    def save_weights(self, model_path, verbose=True):
        self.model.save(model_path)
        if verbose: self.log.info('Saved trained model weight at {} '.format(model_path))
        
    def load_weights(self, model_path, verbose=True):
        self.model.load_weights(model_path)
        if verbose: self.log.info('Load trained model weight at {} '.format(model_path))

            
    def save_history(self, hist_path, history):
        try:
            with open(hist_path, 'w+') as f:
                json.dump(history, f)
        except:
            with open(hist_path, 'w+') as f:
                hist = dict([(ky, np.array(val).astype(np.float).tolist()) for (ky, val) in history.items()])
                json.dump(hist, f)     
            
    def get_callbacks(self, validation_data=None, model_path=None):
        # Callback
        if 'callbacks' in self.network_info['training_info']:
            callback_names = [cb.strip() for cb in self.network_info['training_info']['callbacks'].split(',')]
            callbacks = []
            for idx, callback in enumerate(callback_names):
                if 'EarlyStopping' in callback:
                    callbacks.append(getattr(keras.callbacks, callback)(monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['mode'],
                                                                        patience=int(self.network_info['training_info']['patience']),
                                                                        min_delta=float(self.network_info['training_info']['min_delta']),
                                                                        verbose=1))
                elif 'ModelCheckpoint' in callback:
                    self.best_model_save = True
                    callbacks.append(getattr(keras.callbacks, callback)(filepath=model_path,
                                                                        monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['mode'],
                                                                        save_best_only=True, save_weights_only=False,
                                                                        verbose=0))
                else:
                    try: callbacks.append(getattr(keras.callbacks, callback)())
                    except: pass
                        
        else:
            callbacks = []
        try: batch_size = int(self.network_info['validation_info']['batch_size'])
        except: batch_size = None
        return callbacks
    
    
    def evaluate(self, x, y):
        self.log.info('Evaluation start!')
        trainingtime = time.time()
        evaluation = self.model.evaluate(x, y, batch_size = len(x), verbose=1)
        self.log.info('Evaluation end with time {}!'.format(time.time()-trainingtime))
        self.log.info('Evaluation: {}'.format(evaluation))
        return evaluation
    

class Dense_with_tree(Dense):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tree_weight=None,
                 **kwargs):
        super(Dense_with_tree, self).__init__(units, 
                                              activation=activation,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint,
                                              **kwargs)
        self.tree_weight = K.constant(tree_weight)
    
    def call(self, inputs):
        output = K.dot(inputs, tf.multiply(self.kernel, self.tree_weight))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_weights(self):
        # ref: https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/engine/base_layer.py#L21
        params = self.weights
        weights = K.batch_get_value(params)
        return weights[0]*K.get_value(self.tree_weight), weights[1]
    
    
#####################################################################################################################
#     initializer with phylogenetic tree information class
#####################################################################################################################
class VarianceScaling_with_tree(VarianceScaling):
    ## ref : https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/initializers.py#L155
    def __init__(self, 
                 tree_weight,
                 scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        super(VarianceScaling_with_tree, self).__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)
        self.tree_weight = tree_weight
        
    def __call__(self, shape, dtype=None):
        from keras.initializers import _compute_fans
        
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            init = K.truncated_normal(shape, 0., stddev,
                                      dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            init = K.random_uniform(shape, -limit, limit,
                                    dtype=dtype, seed=self.seed)
        
        return init * self.tree_weight

    def get_config(self):
        return {
#             'tree_weight': self.tree_weight,
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }
    
def glorot_uniform_with_tree(tree_weight, seed=None):
    return VarianceScaling_with_tree(tree_weight, scale=1., mode='fan_avg', distribution='uniform', seed=seed)

def he_normal_with_tree(tree_weight, seed=None):
    return VarianceScaling_with_tree(tree_weight, scale=2., mode='fan_in', distribution='normal', seed=seed)


    
#####################################################################################################################
#     Phylogenetic Regularized Networks
#####################################################################################################################

class PhyloRegularizedNetwork(Base_Network):
    def __init__(self, network_info, tree_info, log, fold=None, num_classes = 1, 
                 tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                 is_covariates=False, covariate_names = None,
                 lvl_category_dict = None,
                 verbose=True, with_level = True):
        super(PhyloRegularizedNetwork,self).__init__(network_info, log)
        if fold != None: self.fold = fold
        else: self.fold = ''
        # self.TB = TensorBoardWrapper_DeepBiome
        # self.TB = TensorBoardWrapper
        print('running PhyloRegularizedNetwork')
        self.num_classes = num_classes
        self.build_model(tree_info,
                         tree_level_list = tree_level_list,
                         is_covariates=is_covariates, covariate_names=covariate_names, 
                         lvl_category_dict = lvl_category_dict,
                         verbose=verbose)
    
    def set_phylogenetic_tree_info(self, tree_info, tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'], 
                                   is_covariates=False, covariate_names = None,
                                   lvl_category_dict = None,
                                   verbose=True, with_level = True):
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Read phylogenetic tree information')
        self.phylogenetic_tree_info = tree_info
        self.tree_level_list = [lvl_name for lvl_name in tree_level_list if lvl_name in self.phylogenetic_tree_info.columns.tolist()]
        self.phylogenetic_tree_info = self.phylogenetic_tree_info[self.tree_level_list]
        # self.tree_level_list = self.phylogenetic_tree_info.columns.tolist()
        if verbose: 
            self.log.info('Phylogenetic tree level list: %s' % self.tree_level_list)
            self.log.info('------------------------------------------------------------------------------------------')
        ### dictionary to save phylogenetic information
        self.phylogenetic_tree_dict = {'Number':{}}
        for i, tree_lvl in enumerate(self.tree_level_list):
            self.phylogenetic_tree_info[tree_lvl] = self.phylogenetic_tree_info[self.tree_level_list[i:]].agg('; '.join, axis=1)
            ### save each level info to dict
            ### numbers of category and each index for taxa
            if lvl_category_dict == None: 
                lvl_category = self.phylogenetic_tree_info[tree_lvl].unique()
                # lvl_category = self.phylogenetic_tree_info[tree_level_list[i:]].drop_duplicates()[tree_lvl]
            else: 
                lvl_category = lvl_category_dict[i]
            lvl_num = lvl_category.shape[0]
            if verbose: self.log.info('    %6s: %d' % (tree_lvl, lvl_num))
            self.phylogenetic_tree_dict[tree_lvl] = dict(zip(lvl_category, np.arange(lvl_num)))
            self.phylogenetic_tree_dict['Number'][tree_lvl]=lvl_num
            if is_covariates and i==len(self.tree_level_list)-1:
                lvl_category = np.append(lvl_category, covariate_names)
                lvl_num = lvl_category.shape[0]
                if verbose: self.log.info('    %6s: %d' % ('%s_with_covariates'%tree_lvl, lvl_num))
                self.phylogenetic_tree_dict['%s_with_covariates' % tree_lvl] = dict(zip(lvl_category, np.arange(lvl_num)))
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Phylogenetic_tree_dict info: %s' % list(self.phylogenetic_tree_dict.keys()))
            self.log.info('------------------------------------------------------------------------------------------')
        self.phylogenetic_tree = copy.deepcopy(self.phylogenetic_tree_info.iloc[:,:])
        for tree_lvl in self.tree_level_list:
            ### overlapping taxa are indexed with same index
            self.phylogenetic_tree[tree_lvl] = self.phylogenetic_tree[tree_lvl].map(self.phylogenetic_tree_dict[tree_lvl])
        self.phylogenetic_tree = np.array(self.phylogenetic_tree)
        ### phylogenetic_tree will have taxonomy as indexes of phylogenetic_tree_info
        
        ### customized weights: 2^-(taxonomic distance)
        self.tree_weight_list = []
        self.tree_weight_noise_list = []
        num_dict = self.phylogenetic_tree_dict['Number']
        for i in range(len(self.tree_level_list)-1):
            if verbose: self.log.info('Build customized edge weights between [%6s, %6s]'%(self.tree_level_list[i],self.tree_level_list[i+1]))
            # lower = self.phylogenetic_tree[:,i]
            # upper = self.phylogenetic_tree[:,i+1]
            n_lower = num_dict[self.tree_level_list[i]]
            n_upper = num_dict[self.tree_level_list[i+1]]

            tree_d = np.zeros((n_lower,n_upper))
            tree_d_n = np.zeros_like(tree_d) + 0.01
            
            for l_idx in range(n_lower):
                lower_taxo = self.phylogenetic_tree[self.phylogenetic_tree[:, i] == l_idx][0]

                for u_idx in range(n_upper):
                    upper_taxo = self.phylogenetic_tree[self.phylogenetic_tree[:, i+1] == u_idx][0]
                    
                    k = i + 1 # searching taxo level

                    while k <= len(self.tree_level_list) - 1:
                        if lower_taxo[k] == upper_taxo[k]: break

                        k += 1

                    tree_d[l_idx, u_idx] = k - i - 1
                
            if with_level == True:
                tree_d[tree_d>0] += i
            tree_w_2 = 2 ** -(tree_d)
                
            self.tree_weight_list.append(tree_w_2)
            self.tree_weight_noise_list.append(tree_w_2)

        if verbose: self.log.info('------------------------------------------------------------------------------------------')

            
    def build_model(self, tree_info, tree_level_list=['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                    is_covariates=False, covariate_names = None, 
                    lvl_category_dict = None,
                    verbose=True):
        # Load Tree Weights
        self.set_phylogenetic_tree_info(tree_info, 
                                        tree_level_list = tree_level_list,
                                        is_covariates=is_covariates, covariate_names=covariate_names, 
                                        lvl_category_dict = lvl_category_dict,
                                        verbose=verbose)
        
        # Build model
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Build network based on phylogenetic tree information')
            self.log.info('------------------------------------------------------------------------------------------')
        
        weight_initial = self.network_info['architecture_info']['weight_initial'].strip()
        bn = self.network_info['architecture_info']['batch_normalization'].strip()=='True'
        dropout_p = float(self.network_info['architecture_info']['drop_out'].strip())
        
        l1_panelty = self.network_info['architecture_info'].get('weight_l1_panelty', None)
        l2_panelty = self.network_info['architecture_info'].get('weight_l2_panelty', None)
        if l1_panelty != None: kernel_regularizer = keras.regularizers.l1(int(l1_panelty))
        elif l2_panelty != None: kernel_regularizer = keras.regularizers.l2(int(l2_panelty))
        else: kernel_regularizer = None
        
        if self.network_info['architecture_info'].get('tree_thrd', 'False').strip() == 'True': tree_thrd = True
        else: tree_thrd = False
        
            
        weight_decay = self.network_info['architecture_info'].get('weight_decay', None)
        if weight_decay != None: weight_decay = weight_decay.strip()
        if weight_decay == 'None': weight_decay = None
        
        
        x_input = Input(shape=(self.tree_weight_list[0].shape[0],), name='input')
        if is_covariates: covariates_input = Input(shape=covariate_names.shape[0:], name='covariates_input')
        l = x_input
        for i, (tree_w, tree_wn) in enumerate(zip(self.tree_weight_list, self.tree_weight_noise_list)):
            bias_initializer='zeros'
            if weight_initial == 'phylogenetic_tree_glorot_uniform': kernel_initializer = glorot_uniform_with_tree(tree_w, seed=123)
            elif weight_initial == 'phylogenetic_tree_he_normal': kernel_initializer = he_normal_with_tree(tree_w, seed=123)
            else: kernel_initializer = weight_initial
                
            if weight_decay == 'phylogenetic_tree': 
                l = Dense_with_tree(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer, tree_weight=tree_wn, name='l%d_dense'%(i+1))(l)
            elif weight_decay == 'phylogenetic_tree_wo_noise': 
                l = Dense_with_tree(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer, tree_weight=tree_w, name='l%d_dense'%(i+1))(l)
            
            if bn: l = BatchNormalization(name='l%d_bn'%(i+1))(l)
            
            l = Activation('relu', name='l%d_activation'%(i+1))(l)
            
            if dropout_p: l = Dropout(dropout_p, name='l%d_dropout'%(i+1))(l)
                
        
        if is_covariates:
            l = Concatenate(name='biome_covariates_concat')([l,covariates_input])
            
        last_h = Dense(max(1,self.num_classes),
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='last_dense_h')(l)
        
        if self.num_classes == 0:
            p_hat = Activation('linear', name='p_hat')(last_h)
        elif self.num_classes == 1:
            p_hat = Activation('sigmoid', name='p_hat')(last_h)
        else: 
            p_hat = Activation('softmax', name='p_hat')(last_h)
        if is_covariates: self.model = Model(inputs=[x_input, covariates_input], outputs=p_hat)
        else: self.model = Model(inputs=x_input, outputs=p_hat)
    
    def get_trained_weight(self):
        kernel_lists =  [l.get_weights()[0] for l in self.model.layers if 'dense' in l.name]
        kernel_lists_with_name = []
        for i in range(len(kernel_lists)):
            try:
                lower_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i]].items()])
                lower_colname = [lower_dict[ky] for ky in range(len(lower_dict))]
                if len(lower_colname) < kernel_lists[i].shape[0]:
                    lower_colname = lower_colname + list(np.arange(kernel_lists[i].shape[0] - len(lower_colname)))
            except:
                lower_colname = np.arange(kernel_lists[i].shape[0])
            try:
                upper_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i+1]].items()])
                upper_colname = [upper_dict[ky] for ky in range(len(upper_dict))]
                if len(upper_colname) < kernel_lists[i].shape[-1]:
                    upper_colname = upper_colname + list(np.arange(kernel_lists[i].shape[-1] - len(upper_colname)))
            except:
                upper_colname = np.arange(kernel_lists[i].shape[-1])
            kernel_lists_with_name.append(pd.DataFrame(kernel_lists[i], columns=upper_colname, index=lower_colname))
        return kernel_lists_with_name
    
    def get_trained_bias(self):
        kernel_lists =  [l.get_weights()[1] for l in self.model.layers if 'dense' in l.name]
        return kernel_lists
    
    def get_tree_weight(self):
        kernel_lists_with_name = []
        for i in range(len(self.tree_weight_list)):
            try:
                lower_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i]].items()])
                lower_colname = [lower_dict[ky] for ky in range(len(lower_dict))]
                if len(lower_colname) < self.tree_weight_list[i].shape[0]:
                    lower_colname = lower_colname + list(range(self.tree_weight_list[i].shape[0] - len(lower_colname)))
            except:
                lower_colname = np.arange(self.tree_weight_list[i].shape[0])
            try:
                upper_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i+1]].items()])
                upper_colname = [upper_dict[ky] for ky in range(len(upper_dict))]
                if len(upper_colname) < self.tree_weight_list[i].shape[-1]:
                    upper_colname = upper_colname + list(range(self.tree_weight_list[i].shape[-1] - len(upper_colname)))
            except:
                upper_colname = np.arange(self.tree_weight_list[i].shape[-1])
            kernel_lists_with_name.append(pd.DataFrame(self.tree_weight_list[i], columns=upper_colname, index=lower_colname))
        return kernel_lists_with_name