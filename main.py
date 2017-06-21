#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:52:10 2017

@author: Home
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale, scale, maxabs_scale, robust_scale
import os

# use functions in from pandas.core.common 

SAVE_DIR = 'pipelines/' # A directory to save pipelines txt files.


def read_pipeline(fname):
    '''
    Read and excute a pipeline file.
    
    Parameters
    ----------
    fname : file name of the pipeline to be read.
    '''        
    
    if os.path.exists(''.join([SAVE_DIR, fname])) == False:
        raise IOError('File \'{0}\' not found'.format( ''.join([SAVE_DIR, fname]) ))
    with open(''.join([SAVE_DIR, fname]), "rb") as fp:
        log_ = pickle.load(fp)      
        temp_obj = Xplorer()
        for command in log_:                
            eval(command.replace('self','temp_obj'))
    return temp_obj

def eval_models(eda_objs, clfs):
    '''
    Uses a given set of classifiers objects to evaluates a given set of pipelines
    and return their CV scores.
    
    Parameters
    ----------
    pipelines_names: list of strings
                names of the pipelines to compare
    eda_objs : list of objects
    clfs     : list of classifiers
    *kwargs : Additional arguments to pass to sikit-learn's cross_val_score 
    '''        

    if isinstance(clfs, list) is False:
        clfs = [clfs]
    acc = []
    for clf_name, clf in clfs:        
        for pipe_name, obj in eda_objs:   
            X, y = obj.df[obj._get_input_features()], obj.df[obj.y]
            cv_score = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring='r2') #neg_mean_squared_error
            acc.append([(clf_name, pipe_name, v) for v in cv_score])
    acc = [item for sublist in acc for item in sublist] # flatten the list of lists
    return acc


def plot_models(accs):
    cv_df = pd.DataFrame(accs, columns=['model','pipeline', 'score'])
    sns.factorplot(x='model',y='score', hue='pipeline', data=cv_df, kind='bar')
    



class Xplorer:
    '''
    Defines an _EDA_ instance given a dataframe (df) and a response column (y)
    
    Parameters
    ----------
    df :pd.DataFrame or a path to a csv file to the feature matrix
        of shape (n_samples, n_features)
    y  :string
        The name of the target column in 'df'.
    **kwargs: Additional arguments to be passed to Panda's read_csv
    '''        

    def __init__(self, df=None, y=None, **kwargs):
        self.log = []
        self.pipelines = {}
        self.read_data(df=df, y=y, **kwargs)

    ###################################################
    # Internal Operations Functions
    ###################################################

    def read_data(self, df=None, y=None, **kwargs):
        '''
        Defines a dataframe (df) and a response column (y)
        
        Parameters
        ----------
        df :pd.DataFrame or a path to a csv file to the feature matrix
            of shape (n_samples, n_features)
        y  :string
            The name of the target column in 'df'.
        '''

        self._validate_params(params_list   = {'df':df,        'y':y},
                              expected_types= {'df':[str,type(None),pd.DataFrame], 'y':[str,type(None)]})        
        
        if type(df) is str: 
            self.df = pd.read_csv(df, **kwargs).copy()
        elif type(df) is pd.DataFrame:
            self.df = df.copy()
        
        if type(df) is str:
            self._log('self.read_data(df=\'{0}\')'.format(df))
        else:
            #self._log('self.read_data(df=df)')
            print('Cannot log read_data properly.')
        if type(y) is str:
            self.set_target(y)
            
    def set_target(self, y):
        '''
        Sets a target variable (y) if it has not been defined.
        
        Parameters
        ----------
        y  :string
            The name of the target column in 'df'.
        '''        
        self._validate_params(params_list   = {'y':y},
                              expected_types= {'y':[str]})                
        self.y = y
        self._log('self.set_target(y=\'{0}\')'.format(y))

    def _log(self, entry):
        '''
        Add a new entry to the local log (maintained in self.log). This 
        function is to be used internally to record all operations on df
        
        Parameters
        ----------
        entry  :string
            of the command applied on 'df'.
        '''            
        self.log.append(entry)

    def is_categorical(self, column):
        '''
        Tests whether a given column name is a categorical column. To be used
        internally.
        
        Parameters
        ----------
        column  :string
            column name.
        
        Returns
        ---------
        a (boolean) True if the given column name is a categorical and False 
        otherwise.
        '''            
        return column.dtype.name == 'category'

    def is_numeric(self, column):
        '''
        Tests whether a given column name is a numerical column. To be used
        internally.
        
        Parameters
        ----------
        column  :string
            column name.
        
        Returns
        ---------
        a (boolean) True if the given column name is a numerical and False 
        otherwise.
        '''            
        return column.dtype.kind in 'if' # (i)nt or (f)loat

    def is_datetime(self, column):
        pass

    def is_text(self, column):
        pass
    
    def _mask_numeric_features(self):
        '''
        Masks numerical columns.
        
        Returns
        ----------
        A (boolean) mask of numerical columns
        '''            
        return [self.is_numeric(self.df[x]) for x in self.df.columns]
    
    def _get_numeric_features(self):
        '''
        Selects only the numerical features in 'df'
        
        Returns
        ----------
        list of names of numerical features
        '''            
        #return self.df._get_numeric_data().columns
        return self.df.columns[self._mask_numeric_features()].tolist()
    
    def _mask_categorical_features(self):
        '''
        Masks categorical columns.
        
        Returns
        ----------
        A (boolean) mask of categorical columns
        '''            
        return [self.is_categorical(self.df[x]) for x in self.df.columns]
    
    def _get_categorical_features(self):
        '''
        Selects only the categorical features in 'df'
        
        Returns
        ----------
        list of names of categorical features
        '''            
        return self.df.columns[self._mask_categorical_features()].tolist()
    
    def _get_all_features(self): # maybe exclude datetime or whatever ?
        '''
        Selects all feature names in 'df'
        
        Returns
        ----------
        list of names of all features
        '''            
        return self.df.columns.tolist()

    def _get_output_feature(self): # y
        '''
        Selects only the output feature in 'df'
        
        Returns
        ----------
        the name of the output feature
        '''            
        self._validate_params(params_list   = {'y':self.y},
                              expected_types= {'y':[str]})        
        return self.y

    def _get_input_features(self):
        '''
        Selects only the input features in 'df'
        
        Returns
        ----------
        list of names of the input features
        '''            
        self._validate_params(params_list   = {'y':self.y},
                              expected_types= {'y':[str,list]})  
        cols_list = self.df.columns.tolist()
        cols_list.remove(self.y)
        return cols_list

    def _validate_params(self, params_list, expected_types):
        '''
        Internal validator of parameter types. This function is used in many
        functions to validate that given parameters are of expected types.
        
        Parameters
        ----------
        params_list   : A dictionary of parameter names and their values.
        expected_types: A dictionary that maps each parameter name to the
                        expected type
        
        Example
        ----------
        The following line could be used to validate that 'df' is either
        a string object, 'None' or pd.DataFrame, while 'y' is either string
        object or 'None'.
        
        > self._validate_params(params_list   = {
                                                'df':df, 'y':y
                                                },
                                expected_types= {
                                                'df':[str,type(None),pd.DataFrame], 
                                                 'y':[str,type(None)]
                                                 })        
        '''            
        
        for key, types in expected_types.items():
            assert type(params_list[key]) in expected_types[key], 'Parameter {0} has to be {1} but got {2}'.format(key,expected_types[key],type(params_list[key]))

    def freeze_pipeline(self, name):
        '''
        Assign a name to the current log. Like taking a snapshot of the df. 
        It only stores a copy of the commands in log
        
        Using this function across many milestores in the EDA process 
        makes it easier to save and restore a previous snapshot of df.
        
        Parameters
        ----------
        name  :string
            A name of the the current log.
        '''            
        pipeline = { "logs": self.log.copy() }
        self.pipelines[name] = pipeline

    def save_pipeline(self, sname, fname):
        '''
        Write a specific pipeline to the desk.
        
        Parameters
        ----------
        sname :string. 
            The name of the pipeline version (used in self.freeze_pipeline).
        fname :string
            The name of the txt file. (you should include the extension)
        '''
        if os.path.exists(SAVE_DIR) == False:
            os.mkdir(SAVE_DIR)
        pipeline_to_save = self.pipelines[sname]["logs"]
        with open(''.join([SAVE_DIR, fname]), "wb") as fp:
            pickle.dump(pipeline_to_save, fp)

    def freeze_and_save_pipeline(self, name):
        self.freeze_pipeline(name)
        self.save_pipeline(name, name)


    ###################################################
    # Data Manipulation Functions
    ###################################################

    def fix_types(self, col_types=None):
        '''
        Changes datatypes of the columns in the current df.
        
        Parameters
        ----------
        col_types : A dictionary of column names followed by their respective
                    types to be used. It accapts all typse supported in
                    Pandas (e.g. 'int', 'float', 'category', 'object', etc)
        '''            
        for col, ntype in col_types.items():
            self.df[col] = self.df[col].astype(ntype)
        self._log('self.fix_types(col_types={0})'.format(col_types))

        
    def impute_nans(self, col=None, method='median'):
        '''
        Imputes columns with missing values (NaNs). If the column is categorical, 
        it uses the most ferquent category. If the column is numerical it 
        uses either mean or median of that column (set by the user in 'method')
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all columns will be used.
        method  :string {'median' (default) , 'mean'} 
                'median' uses the median value to impute the missing values while
                'mean' uses the arthematic mean.
        '''
        self._validate_params(params_list   = {'col':col,        'method':method},
                              expected_types= {'col':[str,list,type(None)], 'method':[str]})        
        
        if type(col) is str: col = [col]
        if col is None: col = self._get_all_features() #self.df.columns.tolist()
                
        for column in col:
            if self.df[column].isnull().sum() > 0:
                if self.is_categorical(self.df[column]) == True:
                    self.df[column].fillna(self.df[column].value_counts().index[0], inplace=True)
                elif self.is_numeric(self.df[column]) == True:
                    if method == 'median':
                        self.df[column].fillna(self.df[column].median(), inplace=True)
                    elif method == 'mean':
                        self.df[column].fillna(self.df[column].mean(), inplace=True)
                    else:
                        raise TypeError('UNSUPPORTED METHOD')
                else:
                    raise TypeError('UNSUPPORTED data TYPE for ', column)
        self._log("self.impute_nans(col={0},method=\'{1}\')".format(col,method))
    
    def replace_outlier(self, col=None, value=np.nan):
        '''
        Detects and replace outliers in given columns with some value. This only works for 
        columns with numerical values. Can support any methods for detecting
        outliers. All outliers detection methods are defined and used internally.
        By default, all columns will be imputed if col is not defined.
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all numerical columns will 
                be used.
        value : float, string or np.nan
            The value to replace outliers with.
        '''            
        
        def std_based_outlier(data, factor=3):        
            return np.abs(data - data.mean()) / data.std() > factor        
        
        
        self._validate_params(params_list   = {'col':col,        'value':value},
                              expected_types= {'col':[str,list,type(None)], 'value':[float]})        

        
        if type(col) is str: col = [col]
        if col is None: col = self._get_numeric_features()
        
        maskout = self.df[col].apply(std_based_outlier)
        for column in col:
            self.df[column].replace(to_replace=self.df[column][maskout[column]], value= value, inplace=True)

        if str(value) == 'nan':
            self._log("self.replace_outlier(col={0},value=np.nan)".format(col))
        else:
            self._log("self.replace_outlier(col={0},value={1})".format(col,value))


    def remove_feature(self, col=None):
        '''
        Drops a given column from the dataframe (df)
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). 
        '''            
        self._validate_params(params_list   = {'col':col},
                              expected_types= {'col':[str,list]})        

        if type(col) is str: col = [col]        
        self.df.drop(col, axis=1, inplace=True)
        self._log("self.remove_feature(col={0})".format(col))
        
            
    def transform_feature(self, col=None, func_str=None, new_col_name=None, addtional_params=None, **kwargs):
        '''
        Uses an given function to apply a transformation on a specific column.
        This only works for columns with numerical values. If new_col_name
        is given, then the new transformed column will be saved in new column.
        Otherwise, it will override the same column. 
        
        In the background, this uses df.apply() which applies a given funciton.
        This supports any arbitary transform function (lambda function) or scaling
        function from sklearn package (see scale_feature()). 
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all numerical columns will 
                be used.
        func_str  : string (of function)
            Function to apply to each column/row. Accepts both lambda functions
            (enclosed within a string, e.g. 'lambda x: x**2') or an user-defined
            function (e.g. 'my_transformer').
        new_col_name : string
            If given, a new column will be created for the transformed column.
        addtional_params : dictionary
            any additional parameters to be used for external functions (like
            sklearn's scaling functions -- see scale_feature).
        **kwargs: additional arguments to be passed to panda's apply
        '''            
        self._validate_params(params_list   = {'col':col,'func_str':func_str,'new_col_name':new_col_name},
                              expected_types= {'col':[str,list,type(None)], 'func_str':[str],'new_col_name':[list,str,type(None)]})        
        
        func = eval(func_str)
        if func.__name__ != '<lambda>' and func.__module__ != 'sklearn.preprocessing.data':            
            raise TypeError('func is not recognized')
        if type(col) is str: col = [col]
        if col is None: col = self._get_numeric_features()
            
        if new_col_name is None: # inplace
            if addtional_params != None:
                self.df[col] = self.df[col].apply(func, **addtional_params, **kwargs)
            else:
                self.df[col] = self.df[col].apply(func, **kwargs)
        else:
            #if type(new_col_name) is str: 
            #    new_col_name = [new_col_name]
            if addtional_params != None:
                self.df[new_col_name] = self.df[col].apply(func, **addtional_params, **kwargs)
            else:
                self.df[new_col_name] = self.df[col].apply(func, **kwargs)
        if type(new_col_name) is str:
            self._log("self.transform_feature(col={0},func_str=\'{1}\',new_col_name=\'{2}\',addtional_params={3})".format(col,func_str,new_col_name,addtional_params))            
        else:
            self._log("self.transform_feature(col={0},func_str=\'{1}\',new_col_name={2},addtional_params={3})".format(col,func_str,new_col_name,addtional_params))
            
        
    
    def scale_feature(self, col=None, scaling=None, scaling_parms=None):
        '''
        Scales a given set  of numerical columns. This only works for columns 
        with numerical values. 
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all numerical columns will 
                be used.
        scaling  : {'zscore', 'minmax_scale' (default), 'scale', 'maxabs_scale', 
                    'robust_scale'}
            User-defined scaling functions can also be used through self.transform_feature
        scaling_parms : dictionary
            any additional parameters to be used for sklearn's scaling functions.
            
        '''            
        self._validate_params(params_list   = {'col':col,'scaling':scaling},
                              expected_types= {'col':[str,list,type(None)], 'scaling':[str,type(None)]})        
        
        if scaling is None: scaling = 'minmax_scale'
        
        if scaling == 'zscore':
            scaling = 'lambda x: (x - x.mean()) / x.std()'
        elif scaling ==  'minmax_scale' and scaling_parms is None:
            scaling_parms = {'feature_range':(0, 1),'axis':0}
        elif scaling ==  'scale' and scaling_parms is None:
            scaling_parms = {'with_mean':True, 'with_std':True,'axis':0}
        elif scaling ==  'maxabs_scale' and scaling_parms is None:
            scaling_parms = {'axis':0}
        elif scaling ==  'robust_scale' and scaling_parms is None:
            scaling_parms = {'with_centering':True, 'with_scaling':True, 'axis':0} # 'quantile_range':(25.0, 75.0), 
        else:
            raise TypeError('UNSUPPORTED scaling TYPE')

        self.transform_feature(col=col, func_str=scaling, addtional_params=scaling_parms)
    
    def encode_categorical_feature(self, col=None):
        '''
        Convert categorical columns into dummy/indicator columns.
            
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all categorical columns will 
                be used.
        '''            
        self._validate_params(params_list   = {'col':col},
                              expected_types= {'col':[str,list,type(None)]})        
        
        if col is None: col = self._get_categorical_features()
        if type(col) is str: col = [col]
        #check if col is atually a categorical column..
        if (np.array([self.is_categorical(self.df[x]) for x in col])).all() == True:
            self.df =  pd.get_dummies(data=self.df, columns=col)
        else:
            print('Some columns in cols are not categorical.')
        new_types = {coln:'category' for coln in self.df.columns[self.df.columns.str.startswith(tuple(col))]}        
        
        self._log("self.encode_categorical_feature(col={0})".format(col))
        self.fix_types(col_types=new_types)
        

    ###################################################
    # Plotting and visualization Functions
    ###################################################

    def print_stats(self):
        '''
        Calculates and prints summary/diagnostic statistics about the dataset.
        
        '''            
        num_observations, num_features = self.df.shape
        #Rows with at leaset N missing value(s):
        rows_nans_count = (self.df.isnull().sum(axis=1)>0).sum()
        rows_nans_perct = rows_nans_count / float(num_observations)
        #Columns with at leaset N missing value(s):
        cols_nans_count = (self.df.isnull().sum(axis=0)>0).sum()
        cols_nans_perct = cols_nans_count / float(num_features)
        count_per_col = self.df.isnull().sum(axis=0)
        perct_per_col = count_per_col / float(num_observations)
        aggr = pd.concat([count_per_col, perct_per_col], axis=1)
        aggr.columns = ['na_count', 'na_percentage']
        tot_na_count = self.df.isnull().sum().sum() 
        tot_na_perct = self.df.isnull().sum().sum() / float(num_observations*num_features)
        type_counts = self.df.dtypes.value_counts()    
        type_table  = pd.DataFrame(type_counts, columns=['Count'])
        type_table.index.name = 'Type'
        
        print(' ')
        print('='*20)
        print('Diagnostic Report:')
        print('='*20)
        print(' ')
        print('-- Type Table --')
        print(type_table)        
        print(' ')
        print('Number of observations:\t', num_observations)
        print('Number of features:\t', num_features)
        print('Observations with at least 1 missing value(s): {0} ({1}%)'.format(rows_nans_count, rows_nans_perct))
        print('Features with at least 1 missing value(s): {0} ({1}%)'.format(cols_nans_count, cols_nans_perct))
        print(' ')
        print('-- Missing Cells Table --')
        print(aggr)
        print(' ')
        print('Total missing cells: {0} ({1}%) '.format(tot_na_count, tot_na_perct))
        print(' ')
        print('-- Detailed Stats --')
        print(' ')
        
        # Loop through columns and show stats
        for column in self.df.columns:
            print(' ')
            print('===== Column: \'{0}\' ====='.format(column))            
            if self.is_categorical(self.df[column]):
                print(self.df[column].value_counts())
            elif self.is_numeric(self.df[column]):
                print(self.df[column].describe())
            else:
                pass
        
    def explore_feature_variation(self, col=None, use_target=False, **kwargs):
        '''
        Produces univariate plots of a given set of columns. Barplots are used
        for categorical columns while histograms (with fitted density functinos)
        are used for numerical columns.
        
        If use_target is true, then the variation of the given set of columns
        with respect to the response variable are used (e.g., 2d scatter 
        plots, boxplots, etc).
        
        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all columns will be used.
        use_target : bool, default False
            Whether to use the target column in the plots.
        **kwargs: additional arguments to be passed to seaborn's distplot or
            to pandas's plotting utilities..
        '''            
        self._validate_params(params_list   = {'col':col},
                              expected_types= {'col':[str,list,type(None)]})        
        

        if type(col) is str: col = [col]
        if col is None: col = self._get_all_features()
        if use_target == False:
            for column in col:
                if self.is_numeric(self.df[column]) == True:
                    plt.figure(column)
                    #sns.despine(left=True)        
                    sns.distplot(self.df[column], color="m", **kwargs) 
                    plt.title(column)
                    plt.tight_layout()            
                    #plt.figure('boxplot')
                    #sns.boxplot(x=self.df[col], palette="PRGn")
                    #sns.despine(offset=10, trim=True)     
                elif self.is_categorical(self.df[column]) == True:            
                    #print self.df[column].describe()
                    plt.figure(column)
                    #sns.despine(left=True)    
                    if len(self.df[column].unique()) > 30:
                        self.df[column].value_counts()[:20][::-1].plot.barh(**kwargs)
                        #top = pd.DataFrame(data=top)
                        #sns.barplot(y=top.index, x=top)                        
                    else:
                        self.df[column].value_counts()[::-1].plot.barh(**kwargs)
                        #sns.countplot(y=self.df[column])                    
                    plt.title(column)
                    plt.tight_layout()
                else:
                    raise TypeError('TYPE IS NOT SUPPORTED')
        else: # use target variable
            for column in col:
                self.explore_features_covariation(col1=column, col2=self.y, **kwargs)
        
    def explore_features_covariation(self, col1=None, col2=None, third_feature=None, **kwargs):
        '''
        Produces bivariate plots of a given pair of columns. 
        
        If third_feature is not None, then the variation of the given pair of columns
        with respect to the third_feature (e.g., by using third_feature as color)
        
        Parameters
        ----------
        col1 : a string of a column name.
        col2 : a string of a column name.
        third_feature : a string of a column name.
        **kwargs: additional arguments to be passed to seaborn's plotting
        '''            
        
        self._validate_params(params_list   = {'col1':col1,'col2':col2,'third_feature':third_feature},
                              expected_types= {'col1':[str],'col2':[str],'third_feature':[str,type(None)]})        
        
        if third_feature is not None:
            assert third_feature in self.df.columns, 'The third variable is not found.'
            
        # sns.jointplot('tip','total_bill',data=tips,kind='resid')        
        if third_feature == None:
            plt.figure(''.join([col1, col2]))
            if self.is_numeric(self.df[col1]) == True and self.is_numeric(self.df[col2]) == True:
                sns.jointplot(x=col1, y=col2, data=self.df, **kwargs)
            elif self.is_categorical(self.df[col1]) == True and self.is_categorical(self.df[col2]) == True:
                pivtoed = self.df[[col1,col2]].pivot_table(values=None, index=col1, columns=col2, aggfunc=np.size, fill_value=0)
                sns.heatmap(pivtoed, annot=True, fmt="d", linewidths=.5, **kwargs)
            else:
                sns.boxplot(x=col1, y=col2, data=self.df, **kwargs)
                # or overlayed density plots...
                #sns.distplot(tips.total_bill[tips.sex=='Male'],label='One', kde=False, ax=ax)
                #sns.distplot(tips.total_bill[tips.sex=='Female'],label='two', kde=False, ax=ax)
#                              marginal_kws=dict(bins=15, rug=False),                 
            
            plt.title(' vs. '.join([col1, col2]))
            #plt.tight_layout()   

        else:
            plt.figure(''.join([col1, col2]))
            
            if self.is_categorical(self.df[third_feature]) == True:
                if self.is_numeric(self.df[col1]) == True and self.is_numeric(self.df[col2]) == True:
                    #sns.jointplot(x=col1, y=col2, kind='reg', data=self.df, 
                    #              marginal_kws=dict(bins=15, rug=True))
                    sns.lmplot(x=col1,y=col2,data=self.df, hue=third_feature, fit_reg=True, ci=None, **kwargs)
                elif self.is_categorical(self.df[col1]) == True and self.is_categorical(self.df[col2]) == True:
                    print('Not supported.')
                    ######### FIX HERE ##########                    
                    #pivtoed = self.df[[col1,col2,third_feature]].pivot_table(values=None, index=col1, columns=col2, aggfunc=np.size, fill_value=0)
                    #sns.heatmap(pivtoed, annot=True, fmt="d", linewidths=.5)
                else: # mixed
                    sns.boxplot(x=col1, y=col2, data=self.df, hue=third_feature, **kwargs)
            elif self.is_numeric(self.df[third_feature]) == True:
                if self.is_numeric(self.df[col1]) == True and self.is_numeric(self.df[col2]) == True:
                    print('Not supported.')
                    #sns.lmplot(x=col1, y=col2, col=third_feature, data=self.df, col_wrap=2, size=3, **kwargs)                   
                elif self.is_categorical(self.df[col1]) == True and self.is_categorical(self.df[col2]) == True:
                    pivtoed = self.df[[col1,col2,third_feature]].pivot_table(values=third_feature, index=col1, columns=col2, aggfunc=np.mean, fill_value=0)
                    #sns.heatmap(pivtoed, annot=True, linewidths=.5)
                    sns.boxplot(x=col1, y=third_feature, data=self.df, hue=col2, **kwargs)
                else: # mixed
                    if self.is_categorical(self.df[col1]) == True:
                        sns.lmplot(x=col2,y=third_feature,data=self.df, hue=col1, fit_reg=True, ci=None, **kwargs)
                    else:
                        sns.lmplot(x=col1,y=third_feature,data=self.df, hue=col2, fit_reg=True, ci=None, **kwargs)                        
            else:
                raise TypeError('Y has unsupported type.')
            
            #plt.title(' vs. '.join([col1, col2]))
            #plt.tight_layout()   
 