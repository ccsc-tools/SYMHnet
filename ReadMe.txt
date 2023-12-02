This ReadMe explains the requirements and getting started to run the SYMH index prediction using the deep and graph neural networks SYMHnet.

Prerequisites:

Python, Tensorflow, and Cuda:
The initial work and implementation of SYHMnet was done using Python version 3.9.7, Python packages specified in the requirement.txt file, and GPU Cuda version cuda_11.4.r11.4.
Therefore, in order to run the default out-of-the-box models to run some predictions, you should use the exact version of Python and its packages. 
Other versions are not tested, but they should work if you have the environment set properly to run deep learning jobs.

Python Packages:
The following python packages and modules are required to run SYHMnet:
joblib==1.3.1
keras==2.8.0
numpy==1.25.0
pandas==1.5.1
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
tensorflow==2.8.0
tensorflow-gpu==2.8.0
tensorflow-probability==0.14.1
tensorboard==2.8.0

To install the required packages, you may use Python package manager "pip" as follow:
1.	Copy the above packages into a text file, i.e., "requirements.txt"
2.	Execute the command:
pip install -r requirements.txt
Note: There is a requirements file already created for you to use that includes all packages with their versions. 
The files are located in the root directory of the SYHMnet tool. 
Note: Python packages and libraries are sensitive to versions. Please make sure you are using the correct packages and libraries versions as specified above.

Note that you may also use conda to install the same packages following conda's direction and steps to install them.

Cuda Installation Package:
You may download and install Cuda v 12.1 from https://developer.nvidia.com/cuda-12-1-0-download-archive

Package Structure:

After downloading or cloning files from github repository: https://github.com/ccsc-tools/SYMHnet 
the SYHMnet tool/package includes the following folders and files:
 
ReadMe.txt                    - this ReadMe file.
requirements.txt              - includes Python required packages for Python version 3.9.7.
models_storms_1min            - includes the models for 1-minute resolution. 
models_storms_5min            - includes the models for 5-minute resolution. 
logs                          - includes the logging information.
data                          - includes a list of SYHMnet data sets that can be used for training, validation, and prediction.
results                       - will include the prediction result file(s).
figures                       - will include the prediction result figures.
 								
Note: The figures are also saved as PNG files which can be viewed individually using PNG viewer in case the figures are not displayed due to any system or environment issues.

SYHMnet_test.py                  - Python program to test/predict a trained model.
SYHMnet_train.py                 - Python program to train a model and save it to the "models" directory.
SYHMnet_plot_results_figures.py  - Python program to redraw the SYHMnet figures from existing predictions that exist in the "results" directory.
Other files are included as utilities files for training and testing.
 
Running a Test/Prediction Task:

To run a test/prediction, you should use the existing data sets from the "data" directory. 
SYHMnet_test.py is used to run the test/prediction. 

Type: python SYHMnet_test.py 

Without any option will produce all the short term 1-6 hour ahead predictions, save the prediction results, save and display the figures.

Type: python SYHMnet_test.py 33 1 4

Provide a storm number, for example 33, from available storms 26 to 42.
Provide a resolution value, for example 1 for 1-minute and 5 for 5-minute resolution.
Provide a number h-hour ahead, for example 4, to produce h=4 hour ahead predictions, save the prediction results, save and display the figures. Available numeric options are: 1,2,3,4,5, or 6.
You may also provide more than one hour, for example: "1,2,5,6" 
to produce 1,2,5, and 6-hour ahead predictions. The list must be within double quotes. 

Running a Training Task:

SYHMnet_train.py is used to run the training. 
Examples to run a training job:

Type: python SYHMnet_train.py 1 4 

Provide a resolution value, for example 1 for 1-minute and 5 for 5-minute resolution.
Provide a number h-hour ahead, for example 4, to produce h=4 hour ahead predictions, and save the model. Available numeric options are: 1,2,3,4,5, or 6.
You may also provide more than one hour, for example: "1,2,5,6" to train 1,2,5, and 6-hour ahead models. The list must be within double quotes.

Running Re-plotting the Graphs Task:

To redraw the graphs using the predictions that are saved in the "results" directory, use SYHMnet_plot_results_figures.py program.
 
Type: python SYHMnet_plot_results_figures.py

Without any option will re-draw all the short term 1-6 hour ahead predictions for 1-minute and 5-minute resolutions, save and display the figures.

Type: python SYHMnet_plot_results_figures.py 1 4

Provide a resolution value, for example 1 for 1-minute and 5 for 5-minute resolution.
Provide a number h-hour ahead, for example 4, to produce h=4 hour ahead predictions, save and display the figures. Available numeric options are: 1,2,3,4,5, or 6.
You may also provide more than one hour, for example: "1,2,5,6" to plot 1,2,5, and 6-hour ahead predictions. The list must be within double quotes.
