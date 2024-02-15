import pandas as pd
import numpy as np

data_ch_top = pd.read_stata('E:\Empirical Asset Pricing via Machine Learning\data\data_ch_top.dta')
characteristics = list(set(data_ch_top.columns).difference({'permno','DATE','sic2','RET','SHROUT','mve0','prc'}))


import time
from tqdm import tqdm
#missing data are replaced by the cross-sectional median.
for ch in tqdm(characteristics):
     data_ch_top[ch] = data_ch_top.groupby('DATE')[ch].transform(lambda x: x.fillna(x.median()))


stdt = 19850101
stdt = pd.to_datetime(str(stdt), format='%Y%m%d')
data_ch_top['DATE'] = pd.to_datetime(data_ch_top['DATE'], format='%Y%m%d')
data_ch_top = data_ch_top[(data_ch_top['DATE']>=stdt)].reset_index(drop=True)
data_ch_top.drop(['SHROUT','mve0','prc'], axis=1, inplace=True)
data_ch_top.drop(['orgcap','bm','bm_ia'], axis=1, inplace=True)
characteristics = list(set(data_ch_top.columns).difference({'permno','DATE','sic2','RET','SHROUT','mve0','prc','orgcap','bm','bm_ia'}))


#Drops column created for the placeholder value 999, to avoid multicollinearity.
sic_dummies = pd.get_dummies(data_ch_top['sic2'].fillna(999).astype(int), prefix='sic').drop('sic_999', axis=1)
data_ch_top_d = pd.concat([data_ch_top,sic_dummies], axis=1)
data_ch_top_d.drop(['sic2'], inplace=True, axis=1)


#Macroeconomic Predictors Data
data_ma = pd.read_csv('E:\Empirical Asset Pricing via Machine Learning\data\PredictorData_Goyal.csv')
stdt, nddt = 19850101, 20211231
data_ma = data_ma[(data_ma['yyyymm']>=stdt//100)&(data_ma['yyyymm']<=nddt//100)].reset_index(drop=True)


# construct predictor
ma_predictors = ['dp_sp','ep_sp','bm_sp','ntis','tbl','tms','dfy','svar']
data_ma['Index'] = data_ma['Index'].str.replace(',','').astype('float64')
data_ma['dp_sp'] = data_ma['D12']/data_ma['Index']
data_ma['ep_sp'] = data_ma['E12']/data_ma['Index']
data_ma.rename({'b/m':'bm_sp'},axis=1,inplace=True)
data_ma['tms'] = data_ma['lty']-data_ma['tbl']
data_ma['dfy'] = data_ma['BAA']-data_ma['AAA']
data_ma = data_ma[['yyyymm']+ma_predictors]
data_ma['yyyymm'] = pd.to_datetime(data_ma['yyyymm'],format='%Y%m')+pd.offsets.MonthEnd(0)
data_ma.head()


def rank_and_scale_groupwise(df, feature, group_column='DATE'):
    # Apply ranking within each group and scale the ranks
    ranked_scaled = df.groupby(group_column)[feature].transform(
        lambda x: 2 * ((x.rank(method='average') - x.rank(method='average').min()) /
                       (x.rank(method='average').max() - x.rank(method='average').min())) - 1)
    return ranked_scaled    #rank对几个sic会产生missing
def interactions(data_ch, data_ma, characteristics, ma_predictors):
    # construct interactions between firm characteristics and macroeconomic predictors
    data = data_ch.copy()
    data_ma_long = pd.merge(data[['DATE']],data_ma,left_on='DATE',right_on='yyyymm',how='left')
    data = data.reset_index(drop=True)
    data_ma_long = data_ma_long.reset_index(drop=True)
    new_cols = {fc+'*'+mp: data[fc]*data_ma_long[mp] for fc in characteristics for mp in ma_predictors}
    data = pd.concat([data, pd.DataFrame(new_cols)], axis=1)
    del new_cols

    exclude_columns = {'permno', 'DATE', 'RET'}
    features = list(set(data.columns).difference(exclude_columns))
    # Apply rank and scale transformation group-wise
    for feature in features:
        data[feature] = rank_and_scale_groupwise(data, feature, 'DATE')
    X = data[features]

    missing_values_count = X.isna().sum().sum()
    print(f"Total missing values before replacement: {missing_values_count}")
    X.fillna(0, inplace=True)   #rank cause missing to some SIC

    y = data['RET']
    print(f"The shape of the data is: {data.shape}")
    return X, y


#Split the Sample into Training Set, Validation Set and Testing Set
stdt_vld = np.datetime64('2010-01-01')
stdt_tst = np.datetime64('2015-01-01')
def trn_vld_tst(data):
    # training
    X_trn, y_trn= interactions(data[data['DATE']<stdt_vld],data_ma[data_ma['yyyymm']<stdt_vld],characteristics,ma_predictors)
    # validation
    X_vld, y_vld = interactions(data[(data['DATE']<stdt_tst)&(data['DATE']>=stdt_vld)],data_ma[(data_ma['yyyymm']<stdt_tst)&(data_ma['yyyymm']>=stdt_vld)],characteristics,ma_predictors)
    # testing
    X_tst, y_tst = interactions(data[data['DATE']>=stdt_tst],data_ma[data_ma['yyyymm']>=stdt_tst],characteristics,ma_predictors)
    return X_trn, X_vld, X_tst, y_trn, y_vld, y_tst

X_trn, X_vld, X_tst, y_trn, y_vld, y_tst  = trn_vld_tst(data_ch_top_d)



# out-of-sample R squared
def R_oos(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted).flatten()
    predicted = np.where(predicted < 0, 0, predicted)
    return 1 - (np.dot((actual - predicted), (actual - predicted))) / (np.dot(actual, actual))


def evaluate_feature_importance(model, X, y, feature_name, feature_names, r2_vld):

    X_modified = X.copy()
    # Identify the primary feature and related features to set to zero
    # Include both the exact feature name and any features that start with the feature name followed by additional characters
    related_features = [fname for fname in feature_names if
                        fname == feature_name or fname.startswith(feature_name + "*")]
    if isinstance(X_modified, pd.DataFrame):
        X_modified[related_features] = 0
    else:  # For numpy arrays, assuming feature_names aligns with columns
        indices = [feature_names.index(fname) for fname in related_features]
        X_modified[:, indices] = 0

    y_pred = model.predict(X_modified)
    score = r2_score(y, y_pred) - r2_vld
    return score



# OLS
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# OLS with all features
OLS = LinearRegression().fit(X_trn,y_trn)
y_trn_pred = OLS.predict(X_trn)
y_vld_pred = OLS.predict(X_vld)
y_tst_pred = OLS.predict(X_tst)
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")
#get relatively important features
feature_importance_scores = {}
feature_names = X_vld.columns.to_list()
for feature in characteristics:
    score = evaluate_feature_importance(OLS, X_vld, y_vld, feature, feature_names, r2_vld)
    feature_importance_scores[feature] = score
sorted_scores = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
top_20_features = sorted_scores[:20]
feature_names, importance_scores = zip(*top_20_features)
# Create the plot
plt.figure(figsize=(10, 10))
plt.barh(range(len(feature_names)), importance_scores, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Diff in R2')
plt.title('Top 20 Most Important Features')
plt.tight_layout()





#Lasso
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

alphas = np.logspace(-4, 1, 30)
# Placeholder for the best performance metrics
best_alpha = None
best_score = np.inf
# Loop over the hyperparameter values to find the best one
for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(X_trn, y_trn)
    y_vld_pred = model.predict(X_vld)
    score = mean_squared_error(y_vld, y_vld_pred)

    if score < best_score:
        best_score = score
        best_alpha = alpha

# Output the best alpha and its performance
print(f"Best alpha: {best_alpha} with validation MSE: {best_score}")
# Retrain model with the best alpha on the training data
best_model = Lasso(alpha=best_alpha)
best_model.fit(X_trn, y_trn)
# Predict on training and validation data
y_trn_pred = best_model.predict(X_trn)
y_vld_pred = best_model.predict(X_vld)
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
# Predict on test data
y_tst_pred = best_model.predict(X_tst)
# Calculate MSE and R-squared for the test data
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")
#get relatively important features
feature_importance_scores = {}
feature_names = X_vld.columns.to_list()
for feature in characteristics:
    score = evaluate_feature_importance(best_model, X_vld, y_vld, feature, feature_names, r2_vld)
    feature_importance_scores[feature] = score
# Optionally, sort the scores to find the most impactful features
sorted_scores = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
top_20_features = sorted_scores[:20]
feature_names, importance_scores = zip(*top_20_features)
# Create the plot
plt.figure(figsize=(10, 10))
plt.barh(range(len(feature_names)), importance_scores, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Diff in R2')
plt.title('Top 20 Most Important Features')
plt.tight_layout()




#Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Define alphas for Ridge
alphas = np.logspace(-4, 5, 40)
# Placeholder for the best performance metrics
best_alpha = None
best_score = np.inf
# Loop over the alpha values to find the best one
for alpha in alphas:
    model = Ridge(alpha=alpha)  # Use Ridge instead of Lasso
    model.fit(X_trn, y_trn)
    y_vld_pred = model.predict(X_vld)
    score = mean_squared_error(y_vld, y_vld_pred)

    if score < best_score:
        best_score = score
        best_alpha = alpha

print(f"Best alpha: {best_alpha} with validation MSE: {best_score}")
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_trn, y_trn)
y_trn_pred = best_model.predict(X_trn)
y_vld_pred = best_model.predict(X_vld)
y_tst_pred = best_model.predict(X_tst)
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")




#Enet
from sklearn.linear_model import ElasticNet
alphas = np.logspace(-4, -1, 20)
l1_ratios = [0.5, 0.6]  # Range from pure L2 regularization to pure L1
best_alpha = None
best_l1_ratio = None
best_score = np.inf

for alpha in alphas:
    for l1_ratio in l1_ratios:
        print(f"Training with alpha={alpha}, l1_ratio={l1_ratio}")
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_trn, y_trn)
        y_vld_pred = model.predict(X_vld)
        score = mean_squared_error(y_vld, y_vld_pred)

        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_l1_ratio = l1_ratio

print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio} with validation MSE: {best_score}") #Best alpha: 0.0002069138, Best l1_ratio: 0.6
best_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
best_model.fit(X_trn, y_trn)
# Predict on training, validation, and test data
y_trn_pred = best_model.predict(X_trn)
y_vld_pred = best_model.predict(X_vld)
y_tst_pred = best_model.predict(X_tst)
# Calculate and print the performance metrics
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")




#Boosted Regression Trees with GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Define the hyperparameter grid
max_depths = [1, 2]
n_estimators_range = [10, 20, 30, 50, 100, 200, 300]
learning_rates = [0.01, 0.1]
best_params = {'max_depth': None, 'n_estimators': None, 'learning_rate': None}
best_score = np.inf
best_model = None
for max_depth in max_depths:
    for n_estimators in n_estimators_range:
        for learning_rate in learning_rates:
            print(f"Training with max_depth={max_depth}, n_estimators={n_estimators}, learning_rate={learning_rate}")
            # Initialize and train the GradientBoostingRegressor with current set of hyperparameters
            model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                              learning_rate=learning_rate, random_state=1)
            model.fit(X_trn, y_trn)
            y_vld_pred = model.predict(X_vld)
            score = mean_squared_error(y_vld, y_vld_pred)

            if score < best_score:
                best_score = score
                best_params['max_depth'] = max_depth
                best_params['n_estimators'] = n_estimators
                best_params['learning_rate'] = learning_rate
                best_model = model
print(f"Best hyperparameters: {best_params} with validation MSE: {best_score}")  #
y_trn_pred = best_model.predict(X_trn)
y_vld_pred = best_model.predict(X_vld)
y_tst_pred = best_model.predict(X_tst)
# Calculate and print the performance metrics
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")




#Boosted Regression Trees with lightBGM
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
# Create LightGBM datasets for training and validation
lgb_train = lgb.Dataset(X_trn, y_trn)
lgb_val = lgb.Dataset(X_vld, y_vld, reference=lgb_train)

max_depths = [1, 2]  # Convert these depths to num_leaves later
n_estimators_range = [10, 20, 50, 100, 200, 300, 500, 800, 1000]
learning_rates = [0.01, 0.1]
best_mse = np.inf
best_params = {'max_depth': None, 'n_estimators': None, 'learning_rate': None}
best_model = None
for max_depth in max_depths:
    for n_estimators in n_estimators_range:
        for learning_rate in learning_rates:
            print(f"Training with max_depth={max_depth}, n_estimators={n_estimators}, learning_rate={learning_rate}")
            # Convert max_depth to num_leaves
            num_leaves = 2 ** max_depth
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'l2',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': 0.1,
                'bagging_fraction': 0.5,
                'bagging_freq': 5,
                'verbose': 0,
                'seed': 0,
                'feature_fraction_seed': 0, #Used when feature_fraction < 1
                'bagging_seed': 0, #for randomly selecting part of the data without resampling. For boosted regression trees the common practice is to select data without replacement, differing from the bootstrap sampling used in bagging ensembles like Random Forests.
                'data_random_seed': 0,
            }
            # Update n_estimators dynamically
            bst = lgb.train(params, lgb_train, num_boost_round=n_estimators, valid_sets=[lgb_train, lgb_val], callbacks = [lgb.early_stopping(stopping_rounds=50)])
            # Predict on validation data to choose the best model
            y_pred_vld = bst.predict(X_vld, num_iteration=bst.best_iteration)
            # Calculate MSE on validation data
            mse_vld = mean_squared_error(y_vld, y_pred_vld)
            if mse_vld < best_mse:
                best_mse = mse_vld
                best_params['max_depth'] = max_depth
                best_params['n_estimators'] = n_estimators
                best_params['learning_rate'] = learning_rate
                best_model = bst
print(f"Best parameters: {best_params} with validation MSE: {best_mse}")
y_trn_pred = best_model.predict(X_trn)
y_vld_pred = best_model.predict(X_vld)
y_tst_pred = best_model.predict(X_tst)
# Calculate and print the performance metrics
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")
#get relatively important features
feature_importance_scores = {}
feature_names = X_vld.columns.to_list()  # Assuming X_vld is a pandas DataFrame
for feature in characteristics:
    score = evaluate_feature_importance(best_model, X_vld, y_vld, feature, feature_names, r2_vld)
    feature_importance_scores[feature] = score
sorted_scores = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
top_20_features = sorted_scores[:20]
feature_names, importance_scores = zip(*top_20_features)
# Create the plot
plt.figure(figsize=(10, 10))
plt.barh(range(len(feature_names)), importance_scores, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Diff in R2')
plt.title('Top 20 Most Important Features')
plt.tight_layout()





#Random Forest
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Parameter ranges
max_depths = [1, 2, 3, 4, 5]
max_features_list = [3, 5, 10, 20, 30, 50]
n_estimators = 300
best_mse = np.inf
best_params = {'max_depth': None, 'max_features': None}
for max_depth in max_depths:
    for max_features in max_features_list:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                   random_state=1)
        rf.fit(X_trn, y_trn)

        y_pred_vld = rf.predict(X_vld)
        mse_vld = mean_squared_error(y_vld, y_pred_vld)

        if mse_vld < best_mse:
            best_mse = mse_vld
            best_params['max_depth'] = max_depth
            best_params['max_features'] = max_features
print(f"Best parameters: {best_params}")
print(f"Best validation MSE: {best_mse}")
# Optionally, here I train the final model on the full dataset (train + validation) with the best parameters
# After tuning hyperparameters, use all training and validation data to estimate parameters
X_full = pd.concat([X_trn, X_vld], axis=0)
y_full = np.concatenate([y_trn, y_vld])
rf_final = RandomForestRegressor(n_estimators=n_estimators, max_depth=best_params['max_depth'],
                                 max_features=best_params['max_features'], random_state=1)
rf_final.fit(X_full, y_full)
# Evaluate the final model on the test set or unseen data
y_trn_pred = rf_final.predict(X_trn)
y_vld_pred = rf_final.predict(X_vld)
y_tst_pred = rf_final.predict(X_tst)
# Calculate and print the performance metrics
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")
#get relatively important features
feature_importance_scores = {}
feature_names = X_vld.columns.to_list()
for feature in characteristics:
    score = evaluate_feature_importance(rf_final, X_vld, y_vld, feature, feature_names, r2_vld)
    feature_importance_scores[feature] = score
# Optionally, sort the scores to find the most impactful features
sorted_scores = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
top_20_features = sorted_scores[:20]
feature_names, importance_scores = zip(*top_20_features)
# Create the plot
plt.figure(figsize=(10, 10))
plt.barh(range(len(feature_names)), importance_scores, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Diff in R2')
plt.title('Top 20 Most Important Features')
plt.tight_layout()




# NN5
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
seed_value = 0
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
# Define L2 regularization strength
l2_strengths = [0.00001, 0.0001, 0.001]
learning_rates = [0.01, 0.01]
best_val_mse = float('inf')
best_settings = {'l1_strength': None, 'learning_rate': None, 'model': None}
for l2_strength in l2_strengths:
    for lr in learning_rates:
        print(f"l2_strength={l2_strength}, lr={lr}")
        model = models.Sequential([
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(1)  # Adjust based on your task
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_trn, y_trn, validation_data=(X_vld, y_vld),
                  epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

        val_mse = model.evaluate(X_vld, y_vld, verbose=0)[1]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_settings['l2_strength'] = l2_strength
            best_settings['learning_rate'] = lr
            best_settings['model'] = model

print(f"Best Validation MSE: {best_val_mse}")
print(f"Best Settings: L2 Strength = {best_settings['l2_strength']}, Learning Rate = {best_settings['learning_rate']}")
# Predicting y_tst using the best model
y_trn_pred = best_settings['model'].predict(X_trn, verbose=0)
y_vld_pred = best_settings['model'].predict(X_vld, verbose=0)
y_tst_pred = best_settings['model'].predict(X_tst, verbose=0)
# Calculate and print the performance metrics
r2_trn = r2_score(y_trn, y_trn_pred)
r2_vld = r2_score(y_vld, y_vld_pred)
mse_tst = mean_squared_error(y_tst, y_tst_pred)
r2_tst = r2_score(y_tst, y_tst_pred)
r2oos_tst = R_oos(y_tst, y_tst_pred)
print(f"Training R-squared: {r2_trn}")
print(f"Validation R-squared: {r2_vld}")
print(f"Test MSE: {mse_tst}")
print(f"Test R-squared: {r2_tst}")
print(f"Out-of-sample R-squared: {r2oos_tst}")






# NN4
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
seed_value = 0
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
# Define L2 regularization strength
l2_strengths = [0.00001, 0.0001, 0.001]  # Example values, adjust based on your needs
learning_rates = [0.01, 0.01]
best_val_mse = float('inf')
best_settings = {'l1_strength': None, 'learning_rate': None, 'model': None}
# You can continue using your loop for hyperparameter tuning
for l2_strength in l2_strengths:
    for lr in learning_rates:
        print(f"l2_strength={l2_strength}, lr={lr}")
        model = models.Sequential([
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
            layers.BatchNormalization(),
            layers.Dense(1)  # Adjust based on your task
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_trn, y_trn, validation_data=(X_vld, y_vld),
                  epochs=100, callbacks=[early_stopping], batch_size=5000, verbose=1)

        val_mse = model.evaluate(X_vld, y_vld, verbose=0)[1]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_settings['l2_strength'] = l2_strength
            best_settings['learning_rate'] = lr
            best_settings['model'] = model

print(f"Best Validation MSE: {best_val_mse}")
print(f"Best Settings: L2 Strength = {best_settings['l2_strength']}, Learning Rate = {best_settings['learning_rate']}")




