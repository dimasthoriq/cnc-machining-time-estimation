# About
This is the code developed for my undergrad's final thesis research, CNC Milling Machining Time Estimation with Artificial Neural Networks. This code is for academic purpose only and the thesis paper is not to be published whatsoever.

The research purpose is to develop a neural network model that are capable to accurately estimate the machining time of a CNC Milling workstation, given machining parameters and project data. Lasso regression is used for comparative reason. A simple GUI app prototype is also developed to be used for production.

I included the notebook file in case you want to view the result direcly from the github page

# Result
* MLR Model with Lasso: 237.35 seconds (Training RMSE) & 293.40 seconds (Validation RMSE)
* ANN Model: 185.44 seconds (Training RMSE) & 166.55 seconds (Validation RMSE)

The ANN model was selected to be deployed in the software prototype for implementation testing

Testing results:
* ANN Model: 196.35 seconds
* CAM Software (Current method): 712.12 seconds
