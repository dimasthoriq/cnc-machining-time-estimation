# About
This is the code developed for below paper published at Procedia CIRP (57th Intl Conference on Manufacturing System):
Anas Ma'ruf, Dimas Ahmad Thoriq, and Kresna Surya Buwana. An Early Machining Time Estimation for Make-to-Order Manufacturing Using Machine Learning Approach. In Procedia CIRP, 130: 106-111, 2024.

The research purpose is to develop a neural network model that are capable to accurately estimate the machining time of a CNC Milling workstation, given machining parameters and project data. Lasso regression model is used for benchmarking reason. A simple GUI app prototype was also developed for production.

The ipynb file was included in case to view the result direcly from the github page

# Abstract
PT X is a make-to-order (MTO) machining product manufacturing company. PT X  needs to estimate each order’s lead time in order to estimate the cost at an early stage of the order cycle, due to the unique nature of orders in the MTO industry. These time and cost estimations would then be used to negotiate their proposed fee to the customer. PT X currently utilizes CAM software to estimate their CNC machining time, which turns out to produce a recognizable figure of deviation from the actual CNC machining time.

This research tries to develop a CNC machining time estimation method using a machine learning approach, an artificial neural network model, to utilize the abundant machining data available in PT X. The development of the CNC machining time estimation model uses an Artificial Neural Network (ANN) model as the proposed model and a Multiple Linear Regression model for benchmarking purpose. The model development adopts a  popular cross-industry standard for data mining projects, the CRISP-DM framework.

The ANN model proved superior in accuracy and reliability against the benchmark model, thus being deployed in the proposed software prototype during the implementation test. The test result using 62 rows of testing data shows that the proposed ANN model is capable of estimating unseen data in PT X quite accurately, recording RMSE of 196.35 seconds with 147,49 seconds of absolute error standard deviation. This level of performance is equal to reducing 72% of the RMSE produced by the current method of estimation in PT X during the implementation test. Several machining parameters such as cut length and stepover showed to be significant towards the CNC machining time.

Keywords: Artificial neural network, machine learning, machining time estimation, CNC machining

# Framework
The diagram below describes the workflow of the system being developed
![Research Framework](https://drive.google.com/uc?id=1dF2sxIQ7ru2ovrg30Fv_Z09rN3tRmpj8)

This flowchart depicts the preprocessing steps to clean the data
![Preprocessing Flowchart](https://drive.google.com/uc?id=1wfkyXSwlRIN3wJPDeRDlaAlgRStbeCXT)

Dataset were split into Training, Validation, and Testing set using below scheme
![Splitting Scheme](https://drive.google.com/uc?id=1iddoPe0qu5vGzk2hZvBsGdEUDzOK4lzm)

# Result
* MLR Model with Lasso: 237.35 seconds (Training RMSE) & 293.40 seconds (Validation RMSE)
* ANN Model: 185.44 seconds (Training RMSE) & 166.55 seconds (Validation RMSE)

The ANN model was selected to be deployed in the software prototype for implementation testing

Testing RMSE:
* ANN Model: 196.35 seconds
* CAM Software (Current method): 712.12 seconds
