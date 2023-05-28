from tkinter import *
from tkinter import ttk, filedialog
import csv
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

global path_csv
global df
global df_cam
global X_train
global X_test
global y_train
global y_test
global path_cam
global epoch_step

np.random.seed(15)
tf.random.set_seed(15)

root = Tk()
root.title('Estimasi Waktu Permesinan Frais CNC')
root.geometry("640x480+460+150")

#Defining Functions
def home():
    for widgets in root.winfo_children():
        widgets.place_forget()

    global machine
    machine = []

    heading_home1.place(relx=0.5, rely=0.1, anchor=N)
    heading_home2.place(relx=0.5, rely=0.2, anchor=N)
    button_train_model.place(relx=0.5, rely=0.5, anchor=N)
    button_estimate_time.place(relx=0.5, rely=0.6, anchor=N)
    button_exit.place(relx=0.5, rely=0.7, anchor=N)

def train_menu():   
    for widgets in root.winfo_children():
        widgets.place_forget()

    heading_trainmenu.place(relx=0.1, rely=0.05, anchor=NW)
    caption_trainmenu1.place(relx=0.1, rely=0.15, anchor=NW)
    caption_trainmenu2.place(relx=0.1, rely=0.35, anchor=NW)
    folder_path_trainmenu.place(relx=0.1, rely=0.4, anchor=NW)
    button_filedir_trainmenu.place(relx=0.85, rely=0.4, anchor=NW)
    button_run_trainmenu.place(relx=0.8, rely=0.85, anchor=NW)
    button_back.place(relx=0.65, rely=0.85, anchor=NW)

def estimate_menu():
    for widgets in root.winfo_children():
        widgets.place_forget()
    
    heading_estmenu.place(relx=0.1, rely=0.05, anchor=NW)
    caption_estmenu1.place(relx=0.1, rely=0.15, anchor=NW)
    caption_estmenu2.place(relx=0.1, rely=0.35, anchor=NW)
    folder_path_estmenu.place(relx=0.1, rely=0.4, anchor=NW)
    button_filedir_estmenu.place(relx=0.85, rely=0.4, anchor=NW)
    button_input_estmenu.place(relx=0.8, rely=0.85, anchor=NW)
    button_back.place(relx=0.65, rely=0.85, anchor=NW)

def load_folder():
    global path_cam
    global df_cam

    folder_path_estmenu.config(state="normal")
    folder_path_estmenu.delete(1.0, END)
    path_cam = filedialog.askdirectory(initialdir="C:/Users/Dimas/Documents/TI/TA/2 Coba PMill/PMill", title="Select CAM folder")
    folder_path_estmenu.insert(END, path_cam)
    folder_path_estmenu.config(state="disable")

    first_object = True
    for current_dir, dirs, files in os.walk(path_cam, topdown = True):
        for d in dirs:
                if "SetupSheets_files" in d:
                    path = os.path.join(current_dir, d)
                    for curr_dir, dirz, filez in os.walk(path, topdown = True):
                        for f in filez:
                            if "template database CSM" in f:
                                p = os.path.join(curr_dir, f)
                                print("Loading File: ", p)
                                try:
                                    table = pd.read_html(p)
                                    table = np.array(table, dtype = object)
                                    table = table.reshape(table.shape[1],table.shape[2])
                                    if first_object:
                                        df_cam = pd.DataFrame(table)
                                        df_cam.columns = df_cam.iloc[0]
                                        df_cam = df_cam.iloc[1:,:43].reset_index(drop = True)
                                        first_object = False
                                    else:
                                        df_temp = pd.DataFrame(table)
                                        df_temp.columns = df_temp.iloc[0]
                                        df_temp = df_temp.iloc[1:,:43].reset_index(drop = True)
                                        df_cam = pd.concat([df_cam, df_temp], ignore_index = True)
                                except:
                                    pass

def load_file():
    global path_csv

    folder_path_trainmenu.config(state="normal")
    folder_path_trainmenu.delete(1.0, END)
    path_csv = filedialog.askopenfilename(initialdir="C:/Users/Dimas/Documents/TI/TA/3 Python", title="Select (.csv) file", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
    folder_path_trainmenu.insert(END, path_csv)
    folder_path_trainmenu.config(state="disable")

def preprocess_data(df):

    caption_progress_trainmenu1.place(relx=0.1, rely=0.65, anchor=NW)
    progressbar_trainmenu.place(relx=0.1, rely=0.7, anchor=NW)
    progressbar_trainmenu["value"] = 0
    progressbar_trainmenu["maximum"] = 120

    if len(df.columns) > 35:
        df = df.dropna(subset = ["Aktual"]).reset_index(drop = True)
        df['Estimasi_Detik'] = df['Total_Time'].apply(lambda row: (dt.strptime(row,'%H:%M:%S').hour)*3600 + 60*(dt.strptime(row,'%H:%M:%S').minute) + dt.strptime(row,'%H:%M:%S').second)
        df['Aktual_Detik'] = df['Aktual'].apply(lambda row: row.hour*3600 + row.minute*60 + row.second)
        df = df.drop(columns = ['Toolpath', 'Project', 'Tool_Name', 'Cutting_Time', 'Total_Time', 'Aktual'])
        df = df.fillna(value = {"Tool_Tip_Radius":0, "Global_Thickness":0, "Radial_Thickness":0, "Axial_Thickness":0, "Stepover":0, "Stepdown":0})
        exception = ['Machine', 'Strategy', 'Tool_Type', 'Holder', 'Estimasi_Detik']
        num_col = []
        for i in df.columns:
            if i not in exception:
                num_col.append(i)
        df_out = df[num_col].to_numpy()
        covariance  = np.cov(df_out , rowvar=False)
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)
        centerpoint = np.mean(df_out , axis=0)
        distances = []
        for i, val in enumerate(df_out):
            p1 = val
            p2 = centerpoint
            distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
            distances.append(distance)
        distances = np.array(distances)
        cutoff = chi2.ppf(0.99, df_out.shape[1])
        safeIndexes = np.where(distances <= cutoff)
        df = df.iloc[safeIndexes].reset_index(drop = True)

    x = df.iloc[:, :-2]
    y = df.iloc[:, -1]
    cat_col = ['Machine', 'Strategy', 'Tool_Type', 'Holder']
    x = pd.get_dummies(x, columns = cat_col, drop_first = True)
    num_col = []
    for i in x.columns:
        if i not in cat_col:
            num_col.append(i)
    minmaxscaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))
    x_minmax = pd.DataFrame(minmaxscaler.fit_transform(x[num_col]), columns = num_col)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_minmax, y, test_size = 0.2, random_state = 15)
    caption_progress_trainmenu1.place_forget()
    caption_progress_trainmenu2.place(relx=0.1, rely=0.65, anchor=NW)
    progressbar_trainmenu["value"] = 20

    return X_train, X_test, y_train, y_test

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global epoch_step
        epoch_step += 1
        progressbar_trainmenu["value"] = epoch_step/2000*80
        progressbar_trainmenu.update()

def train_model():
    global epoch_step
    global path_csv
    df = pd.read_csv(path_csv)
    heading_trainmenu.config(text='Training Model')
    caption_trainmenu1.config(text='Please Wait Until Training Is Done')
    button_run_trainmenu.place_forget()
    button_back.place_forget()
    button_filedir_trainmenu.place_forget()
    button_done_cancel.place(relx=0.8, rely=0.85, anchor=NW)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    epoch_step = 0

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
    
    model.add(tf.keras.layers.Dense(units=1, activation = "relu",
                    use_bias=True,
                    bias_initializer = "he_uniform",
                    kernel_initializer = "he_uniform",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.01),
                    bias_regularizer = tf.keras.regularizers.L1L2(l1=100.0, l2= 1.0),
                    activity_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2= 0.01),
                   ))
    
    model.compile(optimizer = 'adam', loss="mean_squared_error", metrics=["mean_squared_error"])

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)
    mc = tf.keras.callbacks.ModelCheckpoint('model_csm.h5', monitor='val_mean_squared_error', save_best_only=True)
    caption_progress_trainmenu2.place_forget()
    caption_progress_trainmenu3.place(relx=0.1, rely=0.65, anchor=NW)
    model.fit(X_train.values, y_train.values, epochs=2000, batch_size=8, verbose=2, shuffle = False, validation_split = 0.2, callbacks = [mc, stop_early, CustomCallback()])
    heading_trainmenu.config(text='Training Done')
    caption_trainmenu1.place_forget()
    button_done_cancel.config(text = "Done")
    caption_progress_trainmenu3.place_forget()
    y_pred = model.predict(X_test.values)
    skor_pred = sklearn.metrics.r2_score(y_test.values, y_pred)
    caption_progress_trainmenu4.config(text="Training Done, Model Saved with {:.2f} Score.".format(skor_pred))
    caption_progress_trainmenu4.place(relx=0.1, rely=0.65, anchor=NW)
    progressbar_trainmenu["value"] = 120
    
def input_manual():
    global df_cam
    global machine

    nrow = len(df_cam)
    inpmachine = inputmachine.get(1.0, "end-1c")
    if inpmachine in ["PRO", "SPLUS", "F1", "MATRIX"]:
        for i in range(nrow):
            machine.append(inpmachine)
            inputButton.place_forget()
            button_run_estmenu.place(relx=0.8, rely=0.85, anchor=NW)
        lbl.config(text = "Provided Input: \n Machine: "+ str(inpmachine))
    else:
        lbl.config(text = "ERROR! Please Choose Between [PRO, SPLUS, F1, OR MATRIX] For Machine Names")


def menu_input():
    for widgets in root.winfo_children():
        widgets.place_forget()

    global df_cam

    heading_inpmenu.place(relx=0.1, rely=0.05, anchor=NW)
    caption = Label(root, text="Input Machine Data For Project {}:".format(df_cam['Project'][0]), justify=LEFT, font="Helvetica 9")
    inputmachine.place(relx=0.1, rely=0.3, anchor=NW)
    captionmachine.place(relx=0.1, rely=0.25, anchor=NW)
    button_back.place(relx=0.65, rely=0.85, anchor=NW)
    inputButton.place(relx=0.8, rely=0.85, anchor=NW)
    caption.place(relx=0.1, rely=0.2, anchor=NW)
    lbl.place(relx=0.1, rely=0.5, anchor=NW)

def estimate():
    for widgets in root.winfo_children():
        widgets.place_forget()
    global machine
    global df_cam
    global path_cam
    
    cat_col = ['Machine', 'Strategy', 'Tool Type', 'Holder']
    machine = pd.DataFrame(machine)
    df_cam.insert(2,'Machine', machine)
    df_cam = df_cam.fillna(value = {"Tool Tip  Radius":0, "Global  Thickness":0, "Radial  Thickness":0, "Axial  Thickness":0, "Stepover":0, "Stepdown":0})
    df_cam = df_cam.iloc[:, :-1]
    x = df_cam.iloc[:, 2:-2]
    x = x.drop(columns = ['Tool Name'])
    possibilities_machine = ["F1", "MATRIX", "PRO", "SPLUS"]
    possibilities_strategy = ["3D Offset", "Along Corner", "Constant Z", "Drilling", "Interleaved Constant Z", "Multipencil Corner",
                             "Offset Area Clearance", "Offset Flat", "Optimised Constant Z", "Pattern", "Pencil Corner",
                             "Profile Area Clearance", "Raster", "Rotary", "Stitch Corner", "Surface Machine"]
    possibilities_tooltype = ["Ball Nosed", "Drill", "End Mill", "Tip Radiused"]
    possibilities_holder = ["MINI-MINI CHCUK MMC8-90", "MINI-MINI CHCUK MMC8-90 TAPER", "SHRINKFITA2", "dummy toolholder name", "michuck2", "mini chuck"]
    
    exists = x["Machine"].tolist()
    difference = pd.Series([item for item in possibilities_machine if item not in exists])
    target_m = x["Machine"].append(pd.Series(difference))
    target = target_m.reset_index(drop=True)
    dummies = pd.get_dummies(target, columns = ["Machine"], drop_first = True)
    dummies_m = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    
    exists = x["Strategy"].tolist()
    difference = pd.Series([item for item in possibilities_strategy if item not in exists])
    target_s = x["Strategy"].append(pd.Series(difference))
    target = target_s.reset_index(drop=True)
    dummies = pd.get_dummies(target, columns = ["Strategy"], drop_first = True)
    dummies_s = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    
    exists = x["Tool Type"].tolist()
    difference = pd.Series([item for item in possibilities_tooltype if item not in exists])
    target_t = x["Tool Type"].append(pd.Series(difference))
    target = target_t.reset_index(drop=True)
    dummies = pd.get_dummies(target, columns = ["Tool Type"], drop_first = True)
    dummies_t = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    
    exists = x["Holder"].tolist()
    difference = pd.Series([item for item in possibilities_holder if item not in exists])
    target_h = x["Holder"].append(pd.Series(difference))
    target = target_h.reset_index(drop=True)
    dummies = pd.get_dummies(target, columns = ["Holder"], drop_first = True)
    dummies_h = dummies.drop(dummies.index[list(range(len(dummies)-len(difference), len(dummies)))])
    
    x = x.drop(columns = ['Machine', "Tool Type", "Strategy", "Holder"])
    
    for i in possibilities_machine:
        if i not in ["F1"]:
            x[i] = dummies_m[i]
    for i in possibilities_strategy:
        if i not in ["3D Offset"]:
            x[i] = dummies_s[i]
    for i in possibilities_tooltype:
        if i not in ["Ball Nosed"]:
            x[i] = dummies_t[i]
    for i in possibilities_holder:
        if i not in ["MINI-MINI CHCUK MMC8-90"]:
            x[i] = dummies_h[i]

    num_col = []
    for i in x.columns:
        if i not in cat_col:
            num_col.append(i)
    minmaxscaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))
    x = pd.DataFrame(minmaxscaler.fit_transform(x[num_col]), columns = num_col)
    saved_model = load_model('best_model_fin_150523_6pm_2.h5')
    
    pred = saved_model.predict(x.values)
    df_cam.insert(df_cam.shape[1],'Estimasi', pred)
    pred_file = "Prediksi {}.xlsx".format(df_cam["Project"][0])
    path_pred = os.path.join(path_cam, pred_file)
    df_cam.to_excel(path_pred, index = False)

    heading_prediction.place(relx=0.05, rely=0.05, anchor=NW)
    caption_prediction.config(text = "Prediction File ({}) \nShould Be On The Same Folder As The CAM Folder".format(pred_file))
    caption_prediction.place(relx=0.05, rely=0.15, anchor=NW)
    button_est_home.place(relx=0.8, rely=0.85, anchor=NW)

#Widget Home
heading_home1 = Label(root, text="Estimasi Waktu Permesinan", justify=CENTER, font="Helvetica 20 bold")
heading_home2 = Label(root, text="Mesin Frais CNC", justify=CENTER, font="Helvetica 20 bold")
button_train_model = Button(root, text="Train Model", justify=CENTER, font="Helvetica 14", height=1, width=24, command=train_menu)
button_estimate_time = Button(root, text="Estimasi Waktu", justify=CENTER, font="Helvetica 14", height=1, width=24, command=estimate_menu)
button_exit = Button(root, text="Exit", justify=CENTER, font="Helvetica 14", height=1, width=24, command=root.destroy)

#Widget Training Menu
heading_trainmenu = Label(root, text="Select Dataset (.CSV) File", justify=CENTER, font="Helvetica 18 bold")
caption_trainmenu1 = Label(root, text="Dataset will be used for model training.", justify=CENTER, font="Helvetica 9")
caption_trainmenu2 = Label(root, text="File (.csv) Path:", justify=CENTER, font="Helvetica 9 bold", height=1)
folder_path_trainmenu = Text(root, font="Helvetica 9", width=65, height=1)
button_filedir_trainmenu = Button(root, text=" ... ", justify=CENTER, font="Helvetica 7", command=load_file, height=1)
progressbar_trainmenu = ttk.Progressbar(root, orien="horizontal", length=500, mode="determinate")
caption_progress_trainmenu1 = Label(root, text="Loading CSV File...", justify=CENTER, font="Helvetica 9 bold", height=1)
caption_progress_trainmenu2 = Label(root, text="Preprocessing Data...", justify=CENTER, font="Helvetica 9 bold", height=1)
caption_progress_trainmenu3 = Label(root, text="Training Model...", justify=CENTER, font="Helvetica 9 bold", height=1)
caption_progress_trainmenu4 = Label(root, text="Training Done, Model Saved.", justify=CENTER, font="Helvetica 9 bold", height=1)
button_run_trainmenu = Button(root, text="Train Model", justify=CENTER, font="Helvetica 9", height=1, width=9, command=train_model)
button_back = Button(root, text="< Back", justify=CENTER, font="Helvetica 9", height=1, width=9, command=home)
button_done_cancel = Button(root, text="Cancel", justify=CENTER, font="Helvetica 9", height=1, width=9, command=home)

#Widget Estimate Menu
heading_estmenu = Label(root, text="Select Folder Proyek CAM", justify=CENTER, font="Helvetica 18 bold")
caption_estmenu1 = Label(root, text="Dataset will be used for estimating.", justify=CENTER, font="Helvetica 9")
caption_estmenu2 = Label(root, text="Folder CAM Path:", justify=CENTER, font="Helvetica 9 bold", height=1)
folder_path_estmenu = Text(root, font="Helvetica 9", width=65, height=1)
button_filedir_estmenu = Button(root, text=" ... ", justify=CENTER, font="Helvetica 7", command=load_folder, height=1)
caption_progress_estmenu1 = Label(root, text="Creating CSV File...", justify=CENTER, font="Helvetica 9 bold", height=1)
caption_progress_estmenu2 = Label(root, text="Preprocessing Data...", justify=CENTER, font="Helvetica 9 bold", height=1)
button_input_estmenu = Button(root, text="Input", justify=CENTER, font="Helvetica 9", height=1, width=9, command=menu_input)
button_run_estmenu = Button(root, text="Estimasi", justify=CENTER, font="Helvetica 9", height=1, width=9, command=estimate)
heading_prediction = Label(root, text="Succesfully Exported The Prediction!", justify=CENTER, font="Helvetica 18 bold")
caption_prediction = Label(root, text="", justify=LEFT, font="Helvetica 9")
button_est_home = Button(root, text="OK", justify=CENTER, font="Helvetica 9", height=1, width=9, command=home)

#Widget Manual Input Menu
heading_inpmenu = Label(root, text="Input Machine Data", justify=CENTER, font="Helvetica 18 bold")
inputmachine = Text(root, height = 1, width = 25)
captionmachine= Label(root, text="(F1/SPLUS/PRO/MATRIX)", justify=LEFT, font="Helvetica 9")
inputButton = Button(root, text = "Input", command = input_manual, height=1, width=9)
lbl = Label(root, text = "", justify=LEFT)

home()

root.mainloop()