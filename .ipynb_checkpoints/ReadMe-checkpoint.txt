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

Note that you may also use conda to install the same packages, following conda's direction and steps to install them.
A yaml file named envronment.yml is included with the package to install a virtual environment in conda named sysmhnet_env with all the required components. 
To install using the yaml file, run the following command in a conda installed machine:
conda env create -f environment.yml 

Cuda Installation Package:
You may download and install Cuda v 12.1 from https://developer.nvidia.com/cuda-12-1-0-download-archive

Package Structure:

After downloading or cloning files from github repository: https://github.com/ccsc-tools/SYMHnet 
the SYHMnet tool/package includes the following folders and files:
 
ReadMe.txt                    - this ReadMe file.
requirements.txt              - includes Python required packages for Python version 3.9.7.
envronment.yml                - includes Python required packages for Python 3.9.7 for conda installation and/or Binder system
models_storms_1min            - includes the models for 1-minute resolution. 
models_storms_5min            - includes the models for 5-minute resolution. 
logs                          - includes the logging information.
data                          - includes a list of SYHMnet data sets that can be used for training, validation, and prediction.
results                       - will include the prediction result file(s).
figures                       - will include the prediction result figures.
 								
Note: The figures are also saved as PNG files which can be viewed individually using PNG viewer in case the figures are not displayed due to any system or environment issues.

SYHMnet_test.py                  - Python program to test/predict a trained model.
Other files are included as utilities files for training and testing.
 
Running a Test Task:

To run a test, you should use the existing data sets from the "data" directory. 
SYHMnet_test.py is used to run the test. 

Usage:
python SYMHnet_test <storm to test: 26 to 42>  <resolution type: 1|5 > <prediction error True|False> <local view (focus on peak storm time) True|False>
For example to test storm 37 for 5-minute resolution with prediction error for local view:
SYMHnet_test 37 5 True True
