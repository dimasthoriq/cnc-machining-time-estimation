#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as dt, time
from scipy.stats import chi2

import tensorflow as tf
import sklearn
from sklearn.linear_model import Lasso
from keras_tuner import BayesianOptimization, HyperModel
from keras.models import load_model
import statsmodels.api as sm


# In[2]:


np.random.seed(15)
tf.random.set_seed(15)


# In[121]:


#Import Data Files & Integrate To One Source File

first_object = True

for current_dir, dirs, files in os.walk("C:\\Users\\Dimas\\Documents\\TI\\TA\\2 Coba PMill\\PMill", topdown = True):
    for d in dirs:
        if "SetupSheets_files" in d:
            path = os.path.join(current_dir, d)
            for curr_dir, dirz, filez in os.walk(path, topdown = True):
                for f in filez:
                    if "template database CSM" or "template setup sheet" in f:
                        try:
                            p = os.path.join(curr_dir, f)
                            print("Loading File: ", p)
                            table = pd.read_html(p)
                            table = np.array(table, dtype=object)
                            table = table.reshape(table.shape[1],table.shape[2])
                            if first_object:
                                df = pd.DataFrame(table)
                                df.columns = df.iloc[0]
                                df = df.iloc[1:,:43].reset_index(drop = True)
                                first_object = False
                            else:
                                df_temp = pd.DataFrame(table)
                                df_temp.columns = df_temp.iloc[0]
                                df_temp = df_temp.iloc[1:,:43].reset_index(drop = True)
                                df = pd.concat([df, df_temp], ignore_index = True)
                        except:
                            pass


# In[122]:


df


# In[5]:


#Export Integrated File So That It Can Be Combined With The Actual Time and Be Imported Right Away Next Time

df.to_excel("dataset_tambahan.xlsx", index = False)
os.getcwd()


# In[3]:


#Import Integrated & Edited Dataset File

df = pd.read_excel("aktual_tambahan.xlsx")
df


# # Data Preprocessing

# In[4]:


#Define A Function For Checking Variable Type, Number, and Percentage of Missing & Unique Values

def df_summ(df):
    result = pd.DataFrame()
    
    result['Kolom'] = df.columns
    result['Tipe'] = df.dtypes.values
    result['Missing'] = df.isna().sum().values
    result['Missing (%)'] = result['Missing']*100/len(df)
    result['Unik'] = df.nunique().values
    result['Unik (%)'] = result['Unik']*100/len(df)
    
    return result


# In[5]:


df_summ(df)


# In[6]:


#Check For Any Duplicated Records

df.duplicated().sum()


# In[7]:


#Check Descriptional Statistics For Numerical Variables In Dataset

df.describe()


# In[8]:


#Drop Records That Doesnt Have The Dependent/Target Variable (Y) Values

df = df.dropna(subset = ["Aktual"]).reset_index(drop = True)


# In[9]:


#Convert From Timestamp Format (HH:mm:ss) to seconds unit

df['Estimasi_Detik'] = df['Total_Time'].apply(lambda row: (dt.strptime(row,'%H:%M:%S').hour)*3600
                                              + 60*(dt.strptime(row,'%H:%M:%S').minute)
                                              + dt.strptime(row,'%H:%M:%S').second)
df['Aktual_Detik'] = df['Aktual'].apply(lambda row: row.hour*3600 + row.minute*60 + row.second)
df


# In[10]:


#Calculate The Performance Metrics for Current/Existing Method of Machining Time Estimation

print('R2: %.3f' % sklearn.metrics.r2_score(df['Aktual_Detik'], df['Estimasi_Detik']))
print('Mean Absolute Error: %.3f' % sklearn.metrics.mean_absolute_error(df['Aktual_Detik'], df['Estimasi_Detik']))  
print('Mean Squared Error: %.3f' % sklearn.metrics.mean_squared_error(df['Aktual_Detik'], df['Estimasi_Detik']))  
print('Root Mean Squared Error: %.3f' % np.sqrt(sklearn.metrics.mean_squared_error(df['Aktual_Detik'], df['Estimasi_Detik'])))


# In[11]:


#Drop Columns That Are Irrelevant For Time Estimation In Nature (Identity Variables)
#and Columns That Has Close to 100% Missing Values

df = df.drop(columns = ['Toolpath', 'Project', 'Tool_Name', 'Shank_Clearance', 'Holder_Clearance', 'Head_Clearance',
                    'Short_Link_Type', 'Long_Link_Type', 'Default_Link_Type', 'Cutting_Time', 'Total_Time',
                    'Aktual'])


# In[12]:


#Replace The Missing Values of The Remaining Columns with 0 Because It Is Left Unspecified by The Programmer/Planner to Zero

df = df.fillna(value = {"Tool_Tip_Radius":0, "Global_Thickness":0, "Radial_Thickness":0, "Axial_Thickness":0,
                      "Stepover":0, "Stepdown":0})


# In[13]:


#Recheck the dataset after preprocessed (1)

df.info()


# In[14]:


#Recheck the dataset after preprocessed (2)

df_summ(df)


# In[15]:


df


# In[16]:


#Check The Correlation Table for The Numerical Variables

df.corr()


# In[17]:


#Visualize Correlation Table With Heat Map

correlation_matrix = df.corr().round(2)
fig, ax = plt.subplots(figsize = (50,50)) 
sns.heatmap(data=correlation_matrix, annot = True, ax = ax)


# In[18]:


exception = ['Machine', 'Strategy', 'Tool_Type', 'Holder', 'Estimasi_Detik']
num_col = []

for i in df.columns:
    if i not in exception:
        num_col.append(i)
df_out = df[num_col].to_numpy()

# Covariance matrix
covariance  = np.cov(df_out , rowvar=False)

# Covariance matrix power of -1
covariance_pm1 = np.linalg.matrix_power(covariance, -1)

# Center point
centerpoint = np.mean(df_out , axis=0)


# Distances between center point and 
distances = []
for i, val in enumerate(df_out):
    p1 = val
    p2 = centerpoint
    distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
    distances.append(distance)
distances = np.array(distances)

# Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
cutoff = chi2.ppf(0.99, df_out.shape[1])

# Index of outliers
outlierIndexes = np.where(distances > cutoff )
safeIndexes = np.where(distances <= cutoff)

print('--- Index of Outliers ----')
print(outlierIndexes)

df.iloc[outlierIndexes]


# In[19]:


clean_df = df.iloc[safeIndexes].reset_index(drop = True)
clean_df


# In[20]:


df_summ(clean_df)


# In[21]:


#Export The Preprocessed Dataset So That We Could Import It Right Away Next Time Without Rerunning The Preprocessing Steps

clean_df.to_csv("bersih_tambahan.csv", index=False)


# In[3]:


#Import The Preprocessed Dataset

df = pd.read_csv("bersih_tambahan.csv")
df.info()


# In[4]:


#Split The Dataset Into Predictor (X) and Target (Y) Variables

x = df.iloc[:, :-2]
y = df.iloc[:, -1]


# In[25]:


x


# In[26]:


#Check The Variable's Distribution

x.hist(figsize = [15,15])


# In[27]:


#Check The Target Variable's Distribution

y.hist(figsize = [15,15])


# In[5]:


#Convert Categorical Variables To A Dummy (One Hot Encoded) Variables, with The First Level/Class Category Dropped
cat_col = ['Machine', 'Strategy', 'Tool_Type', 'Holder']
x = pd.get_dummies(x, columns = cat_col, drop_first = True)


# In[29]:


x


# In[6]:


#Scale The Numerical Predictor Variable.
#We Use Normalization/MinMax Scaler Because The Variables Seem to Have Non-Gaussian (Normal Distribution)

num_col = []
for i in x.columns:
    if i not in cat_col:
        num_col.append(i)
        
minmaxscaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))
x_minmax = pd.DataFrame(minmaxscaler.fit_transform(x[num_col]), columns = num_col)
x_minmax


# In[37]:


#Split Dataset Into Training and Testing Dataset

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_minmax, y, test_size = 0.2, random_state = 15)


# # Data Modelling

# In[32]:


#Try Model With All 55 Independent Variables

X_train_ols = sm.add_constant(X_train)

regmodel = sm.OLS(y_train, X_train_ols).fit()
regmodel.summary()


# In[33]:


#Evaluate The First Model With 55 Independent Variables
X_test_ols = sm.add_constant(X_test, has_constant='add')
y_pred = regmodel.predict(X_test_ols)

print('R2: %.3f' % sklearn.metrics.r2_score(y_test, y_pred))
print('Adjusted R2: %.3f' % (1 - (1 - sklearn.metrics.r2_score(y_test, y_pred)) * (len(y_pred) - 1) / (len(y_pred) - 55 - 1)))
print('Mean Absolute Error: %.3f' % sklearn.metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error: %.3f' % sklearn.metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error: %.3f' % np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))


# # COBAA LASSO

# In[34]:


#Searching for An Optimal Number of Independent Variable

# step-1: specify range of hyperparameters to tune
hyper_params = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]


# step-2: perform grid search
# 2.1 specify model
lasso = Lasso(max_iter = 10e5)     

# 2.2 call GridSearchCV()
model_cv = sklearn.model_selection.GridSearchCV(estimator = lasso, 
                        param_grid = hyper_params, 
                        scoring= ['r2', 'neg_root_mean_squared_error'],
                        refit = 'neg_root_mean_squared_error',
                        verbose = 1,
                        return_train_score = True)      

# fit the model
model_cv.fit(X_train, y_train)

#Save The Result into A Dataframe
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[35]:


# plotting cv results
fig, ax = plt.subplots(1, 2, figsize = (16,6))

ax[0].plot(cv_results["param_alpha"], cv_results["mean_test_r2"])
ax[0].plot(cv_results["param_alpha"], cv_results["mean_train_r2"])
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('R2 Score')
ax[0].legend(['val', 'train'])

ax[1].plot(cv_results["param_alpha"], -cv_results["mean_test_neg_root_mean_squared_error"])
ax[1].plot(cv_results["param_alpha"], -cv_results["mean_train_neg_root_mean_squared_error"])
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('RMSE')
ax[1].legend(['val', 'train'])


# In[36]:


model_cv.best_params_


# In[20]:


lasso = Lasso(alpha=1, max_iter = 10e5).fit(X_train, y_train)

scoring = {'abs_error': 'neg_mean_absolute_error', 'squared_error': 'neg_mean_squared_error', 'r2':'r2'}

scores = sklearn.model_selection.cross_validate(lasso, X_train.values, y_train.values, scoring=scoring,
                                                return_train_score=True, return_estimator = True)


# In[21]:


print('\nTraining R2: %.3f' % scores['train_r2'].mean())
# print('Training Adjusted R2: %.3f' % (1 - (1 - train_score) * (len(y_test) - 1) / (len(y_test) - coeff_used - 1)))
print('Training Mean Absolute Error: %.3f' % -scores['train_abs_error'].mean())
print('Training Root Mean Squared Error: %.3f' % np.sqrt(-scores['train_squared_error'].mean()))

print('\nValidation R2: %.3f' % scores['test_r2'].mean())
# print('Testing Adjusted R2: %.3f' % (1 - (1 - test_score) * (len(y_test) - 1) / (len(y_test) - coeff_used - 1)))
print('Val. Mean Absolute Error: %.3f' % -scores['test_abs_error'].mean())
print('Val. Root Mean Squared Error: %.3f' % np.sqrt(-scores['test_squared_error'].mean()))


# In[22]:


y_pred = lasso.predict(X_test)

train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)


# In[23]:


coeff_used = np.sum(lasso.coef_!= 0)
print('Features Used: ', coeff_used)
for i in np.where(lasso.coef_ != 0):
    print(X_test.columns[i])

# print('\nTraining R2: %.3f' % train_score)
# print('Training Adjusted R2: %.3f' % (1 - (1 - train_score) * (len(y_test) - 1) / (len(y_test) - coeff_used - 1)))

print('\nTesting R2: %.3f' % test_score)
print('Testing Adjusted R2: %.3f' % (1 - (1 - test_score) * (len(y_test) - 1) / (len(y_test) - coeff_used - 1)))

print('\nTesting Mean Absolute Error: %.3f' % sklearn.metrics.mean_absolute_error(y_test, y_pred))
print('Testing Root Mean Squared Error: %.3f' % np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))


# In[24]:


lasso.coef_


# In[25]:


lasso.intercept_


# In[26]:


#Create Dataframe For The Resulted Time Estimation

result = pd.DataFrame()
result['aktual'] = y_test
result['pred'] = y_pred

result


# In[27]:


#Convert From Seconds Unit Back Into Timestamp (HH:mm:ss) Format

result['Aktual'] = result['aktual'].apply(lambda row: str(datetime.timedelta(seconds=int(row))))
result['Estimasi'] = result['pred'].apply(lambda row: str(datetime.timedelta(seconds=int(row))))

result


# # Neural Network

# ## Tensorflow Keras 

# In[9]:


class MyHyperModel(HyperModel):
    def build(self, hyperparams):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
        for i in range(hyperparams.Int('layers', 1, 5)):
            model.add(tf.keras.layers.Dense(units=hyperparams.Int('units_l' + str(i+1), 2, 128, step = 2, sampling = 'log'),
                            use_bias=True,
                            #                         bias_initializer = hyperparams.Choice("bias_l"+str(i+1), ["glorot_normal", "glorot_uniform","he_normal", "he_uniform", "truncated_normal", "variance_scaling"]),
                            activation=hyperparams.Choice("act_l" + str(i+1), ["relu", "sigmoid"]),
#                             kernel_initializer = hyperparams.Choice("kernel_l"+str(i+1), ["glorot_normal", "glorot_uniform","he_normal", "he_uniform"]),
#                             kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                    l2=hyperparams.Float("L2_K"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                   ),
#                             bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_B"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 ),
#                             activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                      l2=hyperparams.Float("L2_A"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                     )
                           ))
            
            if hyperparams.Boolean('Dropout_l'+str(i+1)):
                model.add(tf.keras.layers.Dropout(rate=hyperparams.Float("droprate_l" + str(i+1), 0.25, 0.5, step = 0.25)))
                
        model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                        use_bias=True,
#                         kernel_initializer = hyperparams.Choice("kernel_o", ["he_normal", "he_uniform"]),
                        #                     bias_initializer = hyperparams.Choice("bias_o", ["he_normal", "he_uniform"]),
#                         kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ko", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                l2=hyperparams.Float("L2_Ko", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                               ),
#                         bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Bo", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                              l2=hyperparams.Float("L2_Bo", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                             ),
#                         activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ao", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_Ao", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 )
                       ))
        
        optim=hyperparams.Choice("optimizer",["adam","rmsprop"])
        model.compile(optimizer = optim, loss="mean_squared_error", metrics=["mean_squared_error"])
        
        return model
    
    def fit(self, hyperparams, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hyperparams.Int("batch_size", min_value=2,max_value=64, step = 2, sampling='log'),
            **kwargs,
        )


# In[10]:


tuner =  BayesianOptimization(hypermodel=MyHyperModel(),
                              objective="val_mean_squared_error",
                              max_trials = 1000,
                              seed = 15,
                              project_name="Coba 5 May 1am",
                              directory = "Coba_nn_rev",
                              overwrite = True
                             )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
tuner.search(X_train.values, y_train.values, epochs=2000, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters

tuner.get_best_hyperparameters()[0].values


# In[11]:


best_model = tuner.get_best_models()[0]
best_model.summary()


# In[24]:


class MyHyperModel(HyperModel):
    def build(self, hyperparams):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
        for i in range(hyperparams.Int('layers', 1, 5)):
            model.add(tf.keras.layers.Dense(units=2**(6-i),
                            use_bias=True,
                            #                         bias_initializer = hyperparams.Choice("bias_l"+str(i+1), ["glorot_normal", "glorot_uniform","he_normal", "he_uniform", "truncated_normal", "variance_scaling"]),
                            activation=hyperparams.Choice("act_l" + str(i+1), ["relu", "sigmoid"]),
#                             kernel_initializer = hyperparams.Choice("kernel_l"+str(i+1), ["glorot_normal", "glorot_uniform","he_normal", "he_uniform"]),
#                             kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                    l2=hyperparams.Float("L2_K"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                   ),
#                             bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_B"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 ),
#                             activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                      l2=hyperparams.Float("L2_A"+str(i+1), 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                     )
                           ))
            
#             if hyperparams.Boolean('Dropout_l'+str(i+1)):
#                 model.add(tf.keras.layers.Dropout(rate=hyperparams.Float("droprate_l" + str(i+1), 0.25, 0.5, step = 0.25)))
                
        model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                        use_bias=True,
#                         kernel_initializer = hyperparams.Choice("kernel_o", ["he_normal", "he_uniform"]),
                        #                     bias_initializer = hyperparams.Choice("bias_o", ["he_normal", "he_uniform"]),
#                         kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ko", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                l2=hyperparams.Float("L2_Ko", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                               ),
#                         bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Bo", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                              l2=hyperparams.Float("L2_Bo", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                             ),
#                         activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ao", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_Ao", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 )
                       ))
        
#         optim=hyperparams.Choice("optimizer",["adam","rmsprop"])
        model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
        
        return model
    
    def fit(self, hyperparams, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hyperparams.Int("batch_size", min_value=2,max_value=64, step = 2, sampling='log'),
            **kwargs,
        )


# In[25]:


tuner =  BayesianOptimization(hypermodel=MyHyperModel(),
                              objective="val_mean_squared_error",
                              max_trials = 500,
                              seed = 15,
                              project_name="Coba 10 May 9am",
                              directory = "Coba_nn_rev",
                              overwrite = True
                             )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
tuner.search(X_train.values, y_train.values, epochs=2000, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters

tuner.get_best_hyperparameters()[0].values


# In[8]:


best_model = tuner.get_best_models()[0]
best_model.summary()


# In[11]:


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
    model.add(tf.keras.layers.Dense(units=64,
                    activation= "relu",
                    use_bias=True,
#                     bias_initializer = "he_uniform",
#                     kernel_initializer = "he_normal",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=1.0, l2=0.001),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 0.1),
#                     activity_regularizer = tf.keras.regularizers.L2(l2=0.1)
                   ))
    
    model.add(tf.keras.layers.Dense(units=32,
                    activation= "relu",
                    use_bias=True,
#                     bias_initializer = "glorot_uniform",
#                     kernel_initializer = "glorot_normal",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
#                     activity_regularizer = tf.keras.regularizers.L2(l2=1.00)
                   ))
    
#     model.add(tf.keras.layers.Dense(units=16,
#                     activation= "relu",
#                     use_bias=True,
#                     bias_initializer = "he_uniform",
#                     kernel_initializer = "he_uniform",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.01),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
#                     activity_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
#                    ))                      
    
#     model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                    use_bias=True,
#                     bias_initializer = "he_uniform",
#                     kernel_initializer = "he_uniform",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=1.00, l2=0.1),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
#                     activity_regularizer = tf.keras.regularizers.L1(l1=0.01)
                   ))
    
    model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
    
    return model


# In[12]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_110523_7am.h5', monitor='val_mean_squared_error', verbose=1, save_best_only=True)

model = build_model()
history = model.fit(X_train.values, y_train.values, epochs=2000, batch_size=8, verbose=2, shuffle = False,
                           validation_split = 0.2, callbacks = [stop_early, mc])


# In[13]:


# Retrieve a list of list results on training and validation data
# sets for each training epoch
rmse = np.sqrt(history.history['mean_squared_error'][:782])
val_rmse = np.sqrt(history.history['val_mean_squared_error'][:782])

# Get number of epochs
epochs = range(len(rmse))

# Plot training and validation loss per epoch
plt.plot(epochs, rmse)
plt.plot(epochs, val_rmse)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Loss (RMSE)')
plt.title('Training and validation loss (rmse)')

print('Best Model: Min Training loss (RMSE) = {:.3f}, Min Validation loss (RMSE) = {:.3f}'
      .format(min(rmse), min(val_rmse)))


# In[22]:


class MyHyperModel(HyperModel):
    def build(self, hyperparams):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
        model.add(tf.keras.layers.Dense(units=hyperparams.Int('unit_1', 32, 256, step = 2, sampling='log'),
                        activation= "relu",
                        use_bias=True,
                        bias_initializer = hyperparams.Choice("bias_l1", ["he_uniform", "he_normal"]),
                        kernel_initializer = hyperparams.Choice("kernel_l1", ["he_uniform", "he_normal"]),
#                         kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K1", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                l2=hyperparams.Float("L2_K1", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                               ),
#                         bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B1", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                              l2=hyperparams.Float("L2_B1", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                             ),
#                         activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A1", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_A1", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 )
                       ))
        
        model.add(tf.keras.layers.Dense(units=hyperparams.Int('unit_2', 8, 64, step = 2, sampling='log'),
                        activation= "relu",
                        use_bias=True,
                        bias_initializer = hyperparams.Choice("bias_l2", ["he_uniform", "he_normal"]),
                        kernel_initializer = hyperparams.Choice("kernel_l2", ["he_uniform", "he_normal"]),
#                         kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K2", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                l2=hyperparams.Float("L2_K2", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                               ),
#                         bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B2", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                              l2=hyperparams.Float("L2_B2", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                             ),
#                         activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A2", 0.001, 10000.0, step = 10, sampling = 'log'),
#                                                                  l2=hyperparams.Float("L2_A2", 0.001, 10000.0, step = 10, sampling = 'log')
#                                                                 )
                       ))
        
        model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                                        use_bias=True,
                                        bias_initializer = hyperparams.Choice("bias_o", ["he_normal", "he_uniform"]),
                                        kernel_initializer = hyperparams.Choice("kernel_o", ["he_uniform", "he_normal"]),
                                        #                         kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ko", 0.001, 10000.0, step = 10, sampling = 'log'),
                                        #                                                                l2=hyperparams.Float("L2_Ko", 0.001, 10000.0, step = 10, sampling = 'log')
                                        #                                                               ),
                                        #                         bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Bo", 0.001, 10000.0, step = 10, sampling = 'log'),
                                        #                                                              l2=hyperparams.Float("L2_Bo", 0.001, 10000.0, step = 10, sampling = 'log')
                                        #                                                             ),
                                        #                         activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ao", 0.001, 10000.0, step = 10, sampling = 'log'),
                                        #                                                                  l2=hyperparams.Float("L2_Ao", 0.001, 10000.0, step = 10, sampling = 'log')
                                        #                                                                 )
                 ))
        
        model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
        
        return model
    
    def fit(self, hyperparams, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=8,
            **kwargs,
        )


# In[23]:


tuner =  BayesianOptimization(hypermodel=MyHyperModel(),
                              objective="val_mean_squared_error",
                              max_trials = 150,
                              seed = 15,
                              project_name="Coba 11 May 7am",
                              directory = "Coba_nn_rev",
                              overwrite = True
                             )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
tuner.search(X_train.values, y_train.values, epochs=2000, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters

tuner.get_best_hyperparameters()[0].values


# In[24]:


best_model = tuner.get_best_models()[0]
best_model.summary()


# In[9]:


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
    model.add(tf.keras.layers.Dense(units=128,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=1.0, l2=0.001),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 0.1),
#                     activity_regularizer = tf.keras.regularizers.L2(l2=0.1)
                   ))
    
    model.add(tf.keras.layers.Dense(units=64,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
#                     activity_regularizer = tf.keras.regularizers.L2(l2=1.00)
                   ))       
    
    model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                    use_bias=True,
                    bias_initializer = "he_uniform",
                    kernel_initializer = "he_uniform",
#                     kernel_regularizer = tf.keras.regularizers.L1L2(l1=1.00, l2=0.1),
#                     bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01),
#                     activity_regularizer = tf.keras.regularizers.L1(l1=0.01)
                   ))
    
    model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
    
    return model


# In[10]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_110523_12pm.h5', monitor='val_mean_squared_error', verbose=1, save_best_only=True)

model = build_model()
history = model.fit(X_train.values, y_train.values, epochs=2000, batch_size=8, verbose=2, shuffle = False,
                           validation_split = 0.2, callbacks = [stop_early, mc])


# In[11]:


# Retrieve a list of list results on training and validation data
# sets for each training epoch
rmse = np.sqrt(history.history['mean_squared_error'][:457])
val_rmse = np.sqrt(history.history['val_mean_squared_error'][:457])

# Get number of epochs
epochs = range(len(rmse))

# Plot training and validation loss per epoch
plt.plot(epochs, rmse)
plt.plot(epochs, val_rmse)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Loss (RMSE)')
plt.title('Training and validation loss (rmse)')

print('Best Model: Min Training loss (RMSE) = {:.3f}, Min Validation loss (RMSE) = {:.3f}'
      .format(min(rmse), min(val_rmse)))


# In[24]:


class MyHyperModel(HyperModel):
    def build(self, hyperparams):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
        model.add(tf.keras.layers.Dense(units=128,
                        activation= "relu",
                        use_bias=True,
                        bias_initializer = "he_normal",
                        kernel_initializer = "he_uniform",
                        kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K1", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                               l2=hyperparams.Float("L2_K1", 0.01, 10000.0, step = 10, sampling = 'log')
                                                              ),
                        bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B1", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                             l2=hyperparams.Float("L2_B1", 0.01, 10000.0, step = 10, sampling = 'log')
                                                            ),
                        activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A1", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                                 l2=hyperparams.Float("L2_A1", 0.01, 10000.0, step = 10, sampling = 'log')
                                                                )
                       ))
        
#         if hyperparams.Boolean('Dropout_l1'):
#             model.add(tf.keras.layers.Dropout(rate=))
        
        model.add(tf.keras.layers.Dense(units=64,
                        activation= "relu",
                        use_bias=True,
                        bias_initializer = "he_normal",
                        kernel_initializer = "he_uniform",
                        kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_K2", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                               l2=hyperparams.Float("L2_K2", 0.01, 10000.0, step = 10, sampling = 'log')
                                                              ),
                        bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_B2", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                             l2=hyperparams.Float("L2_B2", 0.01, 10000.0, step = 10, sampling = 'log')
                                                            ),
                        activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_A2", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                                 l2=hyperparams.Float("L2_A2", 0.01, 10000.0, step = 10, sampling = 'log')
                                                                )
                       ))
        
#         if hyperparams.Boolean('Dropout_l3'):
#             model.add(tf.keras.layers.Dropout(rate=))
        
        model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                        use_bias=True,
                        bias_initializer = "he_uniform",
                        kernel_initializer = "he_uniform",
                        kernel_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ko", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                               l2=hyperparams.Float("L2_Ko", 0.01, 10000.0, step = 10, sampling = 'log')
                                                               ),
                        bias_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Bo", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                             l2=hyperparams.Float("L2_Bo", 0.01, 10000.0, step = 10, sampling = 'log')
                                                            ),
                        activity_regularizer = tf.keras.regularizers.L1L2(l1=hyperparams.Float("L1_Ao", 0.01, 10000.0, step = 10, sampling = 'log'),
                                                                 l2=hyperparams.Float("L2_Ao", 0.01, 10000.0, step = 10, sampling = 'log')
                                                                )
                 ))
        
        model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
        
        return model
    
    def fit(self, hyperparams, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=8,
            **kwargs,
        )


# In[25]:


tuner =  BayesianOptimization(hypermodel=MyHyperModel(),
                              objective="val_mean_squared_error",
                              max_trials = 500,
                              seed = 15,
                              project_name="Coba 11 May 1pm",
                              directory = "Coba_nn_rev",
                              overwrite = True
                             )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
tuner.search(X_train.values, y_train.values, epochs=2000, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters

tuner.get_best_hyperparameters()[0].values


# In[56]:


best_model = tuner.get_best_models()[0]
best_model.summary()


# In[59]:


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
    model.add(tf.keras.layers.Dense(units=128,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2= 1.0),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 10.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=10.0, l2= 10.0)
                   ))
    
    model.add(tf.keras.layers.Dense(units=64,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 1.0),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=1.0, l2= 10.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 0.01),
                   ))          
    
    model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                    use_bias=True,
                    bias_initializer = "he_uniform",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.01),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 1.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 0.01),
                   ))
    
    model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
    
    return model


# In[60]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_fin_110523_4pm.h5', monitor='val_mean_squared_error',
                                        verbose=1, save_best_only=True)

model = build_model()
history = model.fit(X_train.values, y_train.values, epochs=2000, batch_size=8, verbose=2, shuffle = False,
                           validation_split = 0.2, callbacks = [stop_early, mc])


# In[61]:


# Retrieve a list of list results on training and validation data
# sets for each training epoch
rmse = np.sqrt(history.history['mean_squared_error'][:1325])
val_rmse = np.sqrt(history.history['val_mean_squared_error'][:1325])

# Get number of epochs
epochs = range(len(rmse))

# Plot training and validation loss per epoch
plt.plot(epochs, rmse)
plt.plot(epochs, val_rmse)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Loss (RMSE)')
plt.title('Training and validation loss (rmse)')

print('Best Model: Min Training loss (RMSE) = {:.3f}, Min Validation loss (RMSE) = {:.3f}'
      .format(min(rmse), min(val_rmse)))


# In[86]:


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))        
    model.add(tf.keras.layers.Dense(units=32,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2= 1.0),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 10.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=10.0, l2= 10.0)
                   ))
    
    model.add(tf.keras.layers.Dense(units=4,
                    activation= "relu",
                    use_bias=True,
                    bias_initializer = "he_normal",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 1.0),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=1.0, l2= 10.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 0.01),
                   ))          
    
#     model.add(tf.keras.layers.Dropout(rate=0.05))
    
    model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                    use_bias=True,
                    bias_initializer = "he_uniform",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.01),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 1.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 0.01),
                   ))
    
    model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])
    
    return model


# In[92]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_fin_150523_6pm_3.h5', monitor='val_mean_squared_error', verbose=1, save_best_only=True)

model = build_model()
history = model.fit(X_train.values, y_train.values, epochs=2000, batch_size=8, verbose=2, shuffle = False,
                           validation_split = 0.2, callbacks = [stop_early, mc])


# In[93]:


# Retrieve a list of list results on training and validation data
# sets for each training epoch
rmse = np.sqrt(history.history['mean_squared_error'])
val_rmse = np.sqrt(history.history['val_mean_squared_error'])

# Get number of epochs
epochs = range(len(rmse))

# Plot training and validation loss per epoch
plt.plot(epochs, rmse)
plt.plot(epochs, val_rmse)
legend_drawn_flag = True
plt.legend(["Training", "Validation"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Epoch')
plt.ylabel('Loss (RMSE)')
plt.title('Training and validation loss (rmse)')

print('Best Model: Min Training loss (RMSE) = {:.3f}, Min Validation loss (RMSE) = {:.3f}'
      .format(min(rmse), min(val_rmse)))


# In[28]:


model.save('C:\\Users\\Dimas\\Documents\\TI\\TA\\3 Python\\Coba_nn')


# # Final Evaluation and Visualization

# In[30]:


Time_train, time_test = sklearn.model_selection.train_test_split(df.iloc[:,-2:], test_size = 0.2, random_state = 15)
time_train, time_val = sklearn.model_selection.train_test_split(Time_train, test_size = 0.2, random_state = 15)


# In[31]:


print(len(time_train))
print(len(time_val))
print(len(time_test))
print(len(time_val) + len(time_test) + len(time_train))


# In[43]:


x_train, x_val, y_train_, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size = 0.2, random_state = 15)


# In[97]:


# load the saved model
saved_model = load_model('best_model_fin_150523_6pm_2.h5')
print(saved_model.summary())
y_pred_train = saved_model.predict(x_train.values)
y_pred_val = saved_model.predict(x_val.values)
y_pred_test = saved_model.predict(X_test.values)

train_score = saved_model.evaluate(x_train, y_train_)
val_score = saved_model.evaluate(x_val, y_val)
test_score = saved_model.evaluate(X_test, y_test)


# In[98]:


#Evaluating model prediction
y_preds = [y_pred_train, y_pred_val, y_pred_test]
ys = [y_train_, y_val, y_test]

for i in range(3):
    print('\nR2: {:.3f}'.format(sklearn.metrics.r2_score(ys[i], y_preds[i])))
    print('Mean Absolute Error: {:.2f}'.format(sklearn.metrics.mean_absolute_error(ys[i], y_preds[i])))
    print('Mean Absolute Percentage Error: {:.2f} %'.format(sklearn.metrics.mean_absolute_percentage_error(ys[i], y_preds[i])*100))
    print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(sklearn.metrics.mean_squared_error(ys[i], y_preds[i]))))


# In[98]:


#Evaluating the current/existing method used
for i in [time_train, time_val, time_test]:
    print('\nR2: {:.3f}'.format(sklearn.metrics.r2_score(i['Aktual_Detik'], i['Estimasi_Detik'])))
    print('Mean Absolute Error: {:.2f}'.format(sklearn.metrics.mean_absolute_error(i['Aktual_Detik'], i['Estimasi_Detik'])))
    print('Mean Absolute Percentage Error: {:.2f} %'.format(sklearn.metrics.mean_absolute_percentage_error(i['Aktual_Detik'], i['Estimasi_Detik'])*100))
    print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(sklearn.metrics.mean_squared_error(i['Aktual_Detik'], i['Estimasi_Detik']))))


# In[101]:


#Create Dataframe For The Resulted Time Estimation

result_nn = pd.DataFrame()
result_nn['Aktual (Detik)'] = y_test
result_nn['Estimasi (Detik)'] = y_pred_test

result_nn


# In[102]:


#Convert From Seconds Unit Back Into Timestamp (HH:mm:ss) Format

result_nn['Aktual'] = result_nn['Aktual (Detik)'].apply(lambda row: str(datetime.timedelta(seconds=int(row))))
result_nn['Estimasi'] = result_nn['Estimasi (Detik)'].apply(lambda row: str(datetime.timedelta(seconds=int(row))))

result_nn.reset_index(drop=True, inplace=True)
result_nn


# In[103]:


# Plot training and validation loss per epoch
plt.plot(result_nn['Aktual (Detik)'])
plt.plot(result_nn['Estimasi (Detik)'])
# plt.plot(range(40), time_test['Estimasi_Detik'])
legend_drawn_flag = True
plt.legend(["Aktual", "Estimasi"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Data Test Points')
plt.ylabel('Waktu Permesinan CNC Milling (Detik)')
plt.title('Hasil Estimasi Model vs Aktual')


# In[104]:


# Plot training and validation loss per epoch
plt.plot(range(20), y_test[:20])
plt.plot(range(20), y_pred_test[:20])
# plt.plot(range(20), time_test['Estimasi_Detik'][:20])
legend_drawn_flag = True
plt.legend(["Aktual", "Estimasi"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Data Test Points')
plt.ylabel('Waktu Permesinan CNC Milling (Detik)')
plt.title('Hasil Estimasi Model vs Aktual')


# In[108]:


# Plot training and validation loss per epoch
plt.plot(range(42), y_test[20:])
plt.plot(range(42), y_pred_test[20:])
# plt.plot(range(20), time_test['Estimasi_Detik'][20:])
legend_drawn_flag = True
plt.legend(["Aktual", "Estimasi"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Data Test Points')
plt.ylabel('Waktu Permesinan CNC Milling (Detik)')
plt.title('Hasil Estimasi Model vs Aktual')


# In[107]:


# Plot training and validation loss per epoch
plt.plot(range(62), y_test)
plt.plot(range(62), y_pred_test)
# plt.plot(range(40), time_test['Estimasi_Detik'])
legend_drawn_flag = True
plt.legend(["Aktual", "Estimasi"], loc=0, frameon=legend_drawn_flag)
plt.xlabel('Data Test Points')
plt.ylabel('Waktu Permesinan CNC Milling (Detik)')
plt.ylim(0, 4000)
plt.title('Hasil Estimasi Model vs Aktual')


# In[ ]:




