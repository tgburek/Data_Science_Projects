import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold


class TargetEncode(BaseEstimator, TransformerMixin): ## Currently will only handle a *binary* target type
    '''Class keywords:
    - cat_features      {'auto' or list of column names} Categorical features to Target Encode
    - lambda05          {int, 'mean', 'median'} The number of counts at which the smoothing factor, lambda, equals 0.5
    - flattening_factor {float} The factor controlling how quickly lambda goes from 0 to 1 around "lambda05"
    - cv                {int} The number of folds for the cross-fitting in "fit_transform"
    - shuffle           {bool} Whether to shuffle the data in "fit_transform" *before* splitting into folds
    - random_state      {int or None} When "shuffle" is True, "random_state" affects the ordering of the indices
                        which controls the randomness of each fold. Pass an "int" for reproducible output
    '''
    def __init__(
        self, 
        cat_features = 'auto', 
        lambda05 = 1,          
        flattening_factor = 1, 
        cv = 5,                
        shuffle = True,        
        random_state = None    
                               
    ):
        self.cat_features = cat_features
        self.lambda05 = lambda05
        self.flattening_factor = float(flattening_factor)
        self.cv = cv ## k folds
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y): 
        '''Fit one encoding on the entire training set'''
        self._fit_encodings_full_train(X, y)
        return self
    
    def transform(self, X): 
        '''Transform the categorical features in X with the "full training set" encodings'''
        X_copy = X.copy()

        X_copy = self._transform_X_cats(X_copy, full_train=True)
        
        ## Convert the DataFrame to an ndarray to be consistent with scikit-learn's TargetEncoder
        X_copy_arr = X_copy.to_numpy(dtype=np.float64)
        
        return X_copy_arr
    
    def fit_transform(self, X, y): 
        '''Fit and transform the training data via the "cross-fitting" strategy'''
        X_copy = X.copy()
        ## Finding the "full training set" encodings which will be saved to self.encodings_ for use with the "transform" method
        ## Also establishes class attributes pertaining to the whole training set
        X_cat_feat_only = self._fit_encodings_full_train(X, y)
        
        ## Set up StratifiedKFold (since we have a binary target) to later cross-fit (n_splits = cv = k)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)
        
        X_copy = X.copy()
        
        ## Return the training and test folds' indices to split X and y into training and test sets for cross-fitting
        for train_ind, test_ind in skf.split(X, y):
            #print(train_ind, test_ind)
            ## "traindf" contains both the features of the training set ("X_train") and the target of the training set ("y_train")
            ## As in "X_cat_feat_only," the target values make up the column "target" in "traindf"
            traindf = X_cat_feat_only.loc[train_ind, :]
            
            ## Fitting the encodings with the present training set of k-1 folds
            encodings, _ = self._fit_encodings(traindf, full_train=False)
            #print(encodings)
            X_copy = self._transform_X_cats(X_copy, full_train=False, df_train=traindf, transform_indices=test_ind, encodings=encodings)
        
        ## Convert the DataFrame to an ndarray to be consistent with scikit-learn's TargetEncoder
        X_copy_arr = X_copy.to_numpy(dtype=np.float64) 
        
        return X_copy_arr
        
    def _fit_encodings_full_train(self, X, y): 
        '''Fit one encoding on the entire training set and set class attributes based on entire training set'''
        if self.cat_features == 'auto':
            ## Retrieves column names of columns with "dtype=object" and converts the result to a list
            self.cat_features_ = X.columns[X.dtypes == type(object)].tolist() 
        else:
            self.cat_features_ = self.cat_features ## Uses the user-supplied list of categorical column names
        
        if type(self.lambda05) == int:
            self.lambda05_ = self.lambda05
        else:
            self.lambda05_ = {} ## "lambda05" will be based on the mean or median count of category instances in a feature
                                ## The dictionary will have the feature as the key and the mean or median count as the value
        
        self.flattening_factor_ = self.flattening_factor ## Makes sense to have both smoothing variables stored this way
        self.classes_ = y.unique() ## The array of unique classes in the target column (0 and 1 for binary)
        self.target_mean_ = np.mean(y) ## The mean of the target for the full training set
        self.categories_ = {}  ## Will store a dictionary of arrays where each array (value) 
                               ## gives the unique categories of a corresponding feature (key)
        
        tempdf = X.loc[:, self.cat_features_].copy() ## Creates a temporary DataFrame using only the categorical features
        tempdf['target'] = y ## Adds the target column to the temporary DataFrame
        
        ## Fit encodings on the full training dataset
        self.encodings_, self.lambdas_ = self._fit_encodings(tempdf, full_train=True)
            
        return tempdf
        
    def _fit_encodings(self, df_train, full_train=True): 
        '''Base fitting function for use with the "cross-fitting" strategy or the entire training set.
        - "df_train" is the DataFrame of training samples.
        - "full_train" indicates whether "df_train" is the entirety of the training samples (True)
        or k-1 folds in a "cross-fitting" strategy (False).
        '''
        
        lambdas = {} ## Will store a dictionary of dictionaries where each feature (key) has a dictionary (value) 
                     ## of smoothing factors, lambdas (sub-value), corresponding to the categories (sub-key) of the feature
        
        encodings = {} ## Will store a dictionary of dictionaries where each feature (key) has a dictionary (value) 
                       ## of encodings (sub-value) corresponding to the categories (sub-key) of the feature
        
        for feature in self.cat_features_:
            ## Group the DataFrame by the categories in "feature" and calculate the group's "target mean" and "length (counts)"
            cat_means = df_train.groupby(feature).agg({'target': (np.mean, len)}).droplevel(0, axis=1)
                
            if full_train: ## If entire training set is being fit
                ## Store the feature and associated categories in the "categories" dictionary
                self.categories_[feature] = cat_means.index.to_numpy()
                y_train_mean = self.target_mean_
            else:
                ## The target (y) mean for the k-1 training folds in the "cross-fitting" strategy
                y_train_mean = df_train['target'].mean()
                    
            ## Store the "lambda05" value associated with this feature
            if self.lambda05 == 'mean':
                lambda05 = cat_means['len'].mean()
                if full_train:
                    self.lambda05_[feature] = lambda05
            elif self.lambda05 == 'median':
                lambda05 = cat_means['len'].median()
                if full_train:
                    self.lambda05_[feature] = lambda05
            else:
                lambda05 = self.lambda05
            
            ## Computing the smoothing factors (lambdas) for each category. Returns a Series
            smooth_factors = 1. / (1. + np.exp(-(cat_means['len'] - lambda05) / self.flattening_factor_))
            
            if full_train:
                lambdas[feature] = smooth_factors.to_dict()
                
            ## Computing the encodings for the categories of the feature
            encodings[feature] = (y_train_mean * (1. - smooth_factors) + cat_means['mean'] * smooth_factors).to_dict()
                
        return encodings, lambdas
    
    
    def _transform_X_cats(self, X_copy, full_train=True, df_train=None, transform_indices=None, encodings=None):
        '''Base transforming function for use after "cross-fitting" or after fitting the entire training set.
        - "full_train" indicates whether the entirety of the training samples were fit (True)
        or whether k-1 folds in a "cross-fitting" strategy were fit (False).
        If "full_train=False," "df_train," "transform_indices," and "encodings" must be supplied.
        If "full_train=True," the only necessary DataFrame is "X_copy", and any other variables will come from "self."
        - "df_train" is the DataFrame of training samples used during cross-fitting (k-1 folds).
        - "transform_indices" are the indices (samples) of "X_copy" to be encoded (given by the test splits of "X" 
        established during cross-fitting).
        - "encodings" are the encodings found when cross-fitting.
        '''
        
        if full_train: ## If entire training set was fit
            transform_indices = list(range(len(X)))
            y_train_mean = self.target_mean_
            encodings = self.encodings_
        else:
            ## The target (y) mean for the k-1 training folds in the "cross-fitting" strategy
            y_train_mean = df_train['target'].mean()
            
        for feature in self.cat_features_:
            ## Determining the categories (if any) within the feature that were *not* seen during "fit"
            ## These categories are encoded with the mean of the target for the full training set
            unknown_value = {category: y_train_mean for category in X_copy[feature].unique()
                             if category not in encodings[feature].keys()
                            }
            if len(unknown_value) > 0:
                X_copy.loc[transform_indices, feature] = X_copy.loc[transform_indices, feature].replace(unknown_value)
            #print(unknown_value)   
            X_copy.loc[transform_indices, feature] = X_copy.loc[transform_indices, feature].replace(encodings[feature])
            #display(X_copy.loc[transform_indices, feature])
        return X_copy  






class ModifiedTargetEncoder(BaseEstimator, TransformerMixin):
    '''Add notes here
    '''
    def __init__(
        self,
        categories = 'auto',
        target_type = 'auto',
        smooth = 'by_feature',
        fit_feature_smooth_factor = None,
        avg_smooth_factors = None,
        cv = 5,
        shuffle = True,
        random_state = None
    ):
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.fit_feature_smooth_factor = fit_feature_smooth_factor
        self.avg_smooth_factors = avg_smooth_factors
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        '''Add notes here'''
        self.target_var_ = np.var(y)
        self.target_mean_ = np.mean(y)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = X.columns.to_numpy()
        self.smoothing_factors_ = {}
        
        tempdf = X.copy()
        tempdf['target'] = y
        
        if self.smooth == 'by_feature':
            if self.fit_feature_smooth_factor is not None:
                for feature in self.fit_feature_smooth_factor:
                    self._fit_smoothing_factor(tempdf, feature)
            else:
                raise ValueError(
                    '''"smooth" keyword is set to "by_feature"
                    but no feature names were provided to "fit_feature_smooth_factor" keyword
                    '''
                )
            
            if self.avg_smooth_factors is not None:
                for feature in self.avg_smooth_factors:
                    self.smoothing_factors_[feature] = np.mean(
                        [v for (k,v) in self.smoothing_factors_.items() if k in self.fit_feature_smooth_factor]
                    )
                    
                auto_features = [x for x in self.feature_names_in_ 
                                 if x not in self.fit_feature_smooth_factor + self.avg_smooth_factors
                                ]
            else:
                auto_features = [x for x in self.feature_names_in_ if x not in self.fit_feature_smooth_factor]
                
            if len(auto_features) > 0:
                for feature in auto_features:
                    self.smoothing_factors_[feature] = 'auto'
                    
        elif isinstance(self.smooth, list):
            if len(self.smooth) != self.n_features_in_:
                raise IndexError(
                    '''The number of smoothing factors provided does not match
                    the number of features (columns) in the X DataFrame
                    '''
                )
                    
            for feature, m in zip(self.feature_names_in_, self.smooth):
                self.smoothing_factors_[feature] = m
                
        else: ## If self.smooth is an integer or "auto"
            for feature in self.feature_names_in_:
                self.smoothing_factors_[feature] = self.smooth
                
        self.encodings_  = []
        self.categories_ = []
        
        self.te = []
        
        for i, feature in enumerate(self.feature_names_in_):
            self.te += [TargetEncoder(
                categories=self.categories, 
                target_type=self.target_type, 
                smooth=self.smoothing_factors_[feature], 
                cv=self.cv, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            ).set_output(transform='default') ## Ensure the output is an ndarray
            ]
            
            self.te[i].fit(X.loc[:,[feature]], y)
            
            self.encodings_  += self.te[i].encodings_
            self.categories_ += self.te[i].categories_
            
            if i == 0:
                self.target_type_ = self.te[0].target_type_
                self.classes_ = self.te[0].classes_
        
        return self
    
    def transform(self, X):
        '''Add notes here'''
        output_ndarray = np.empty((X.shape[0], self.n_features_in_), dtype=np.float64)
        
        for i, feature in enumerate(self.feature_names_in_):
            output_ndarray[:,i] = self.te[i].transform(X.loc[:,[feature]]).flatten()
            
        return output_ndarray
    
    def fit_transform(self, X, y):
        '''Add notes here'''
        self.fit(X, y)
        
        output_ndarray = np.empty((X.shape[0], self.n_features_in_), dtype=np.float64)
        
        self.te = []
        
        for i, feature in enumerate(self.feature_names_in_):
            self.te += [TargetEncoder(
                categories=self.categories, 
                target_type=self.target_type, 
                smooth=self.smoothing_factors_[feature], 
                cv=self.cv, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            ).set_output(transform='default') ## Ensure the output is an ndarray
            ]
            
            output_ndarray[:,i] = self.te[i].fit_transform(X.loc[:,[feature]], y).flatten()
            
        return output_ndarray

    def get_feature_names_out(self, input_features=None):
        '''
        As in the scikit-learn TargetEncoder, "input_features" is not used.
        It is present for "API consistency by convention."
        As in the scikit-learn TargetEncoder, the output feature names use "feature_names_in_."
        '''

        return self.feature_names_in_
    
    def _fit_smoothing_factor(self, df, feature):
        '''Add notes here'''
        feature_groupby = (df
                           .groupby(feature)
                           .agg({'target': (len, np.var)})
                           .droplevel(0, axis=1)
                          )
        
        feature_groupby = feature_groupby[(~feature_groupby['var'].isna()) &
                                          (feature_groupby['var'] != 0.)
                                         ]
        
        smooth_bayes_est = feature_groupby['var'] / self.target_var_
        feature_groupby['lambda'] = self._calc_shrinkage_factor(feature_groupby['len'], smooth_bayes_est)
        
        smooth_opt, _ = curve_fit(self._calc_shrinkage_factor, feature_groupby['len'], feature_groupby['lambda'])
        
        self.smoothing_factors_[feature] = smooth_opt[0]
        
        return self
    
    def _calc_shrinkage_factor(self, nsamples, smooth):
        '''Add notes here'''
        return nsamples / (nsamples + smooth)