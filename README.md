# Prediction of the SYM-H Index Using a Bayesian Deep Learning Method with Uncertainty Quantification

## Author
### Yasser Abduallah, Jason T. L. Wang, Khalid A. Alobaid, Haimin Wang1, Vania K. Jordanova, Vasyl Yurchyshyn, Huseyin Cavus, and Ju Jing

## Abstract
We propose a novel deep learning framework, named SYMHnet, which employs a graph neural network and 
a bidirectional long short-term memory network to cooperatively learn patterns from 
solar wind and interplanetary magnetic field parameters
for short-term
forecasts of the SYM-H index based on
1-minute and 5-minute resolution data. 
SYMHnet takes, as input, the time series of the parameters' values
provided by NASA's Space Science Data Coordinated Archive
and predicts, as output, 
the SYM-H index value
at time point t + w hours
for a given time point t 
where w is 1 or 2.
By incorporating Bayesian inference into the learning framework, 
SYMHnet can quantify both aleatoric (data) uncertainty and
epistemic (model) uncertainty when predicting future SYM-H indices.
Experimental results show that
SYMHnet works well at quiet time and storm time,
for both 1-minute and 5-minute resolution data.
The results also show that
SYMHnet generally performs better than related machine learning methods.
For example, SYMHnet achieves a forecast skill score (FSS) of
0.343
compared to the FSS of 0.074 of a recent gradient boosting machine (GBM) method
when predicting SYM-H indices (1 hour in advance) 
in a large storm (SYM-H = -393 nT) using 5-minute resolution data.
When predicting the SYM-H indices (2 hours in advance)
in the large storm,
SYMHnet achieves an FSS of
0.553 compared to the FSS of
0.087
of the GBM method.
In addition, SYMHnet can provide 
results for both data and model uncertainty quantification, 
whereas the related methods cannot.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### run_SYMHnet.ipynb (Jupyter Notebook for SYMHnet)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/SYMHnet/HEAD?labpath=run_SYMHnet.ipynb)

__Binder Notes__

* Starting Binder might take some time to create and start the image.

* The execution time in Binder varies based on the availability of resources. The average time to run the notebook is 10-15 minutes, but it could be more.

* Binder does not provide GPU docker images. 

* It is recommended to download or clone the GitHub repository and run the tool locally. To clone the repository:<br>
```
git clone git@github.com:ccsc-tools/SYMHnet.git
``` <br>
Direct download like is https://github.com/ccsc-tools/SYMHnet/archive/refs/heads/main.zip


## Installation on local machine
Requires Python==3.9.x (was trained and tested on 3.9.7)

* Run pip install -r requirements.txt (the file is provided within the package)<br>
* You may also use the environment.yml file to create conda virtual environment with all required packages by exeucting the following command:<br>
```
conda env create -f environment.yml 
```
* Manually install the following packages and specified versions:

|Library | Version   | Description  |
|---|---|---|
|keras| 2.8.0 | Deep learning API|
| matplotlib | 3.7.2 | Graphical and visualization tool|
|numpy| 1.25.0 | Array manipulation|
| pandas| 1.5.1 | Data loading, analysis, and manipulation tool|
|scikit-learn| 1.3.0 | Machine learning tool API|
| seaborn | 0.12.2 | Figures visualization look and feel|
| tensorboard| 2.8.0| Provides the visualization and tooling needed for machine learning|
| tensorflow| 2.8.0| Machine learning platform tool |
| tensorflow-gpu| 2.8.0| Deep learning tool for high performance computation |
