# Data2Text - Model for WebNLG 2020+ data set and AX generated data set

This repository contains the code that was used for training and testing the systems described in the masters thesis "A Corpus Construction Framework
for the Data-to-Text Task".

The directories first_manual_eval/, second_manual_eval/ and human_eval/ contain the data that was used for the evaluations of the systems.

The directory predictions/ contains the predictions of the systems on the test sets of all three datasets used in the thesis.

The directory src/ contains the source code.

To create the data files needed for the WebNLG data set, first you have to clone the following git repository into the main directory:

    git clone https://gitlab.com/shimorina/webnlg-dataset.git
    
To get the AX data you have to clone the following git repository:

    git clone https://github.com/Pavlos-96/ECOM-A-German-English-E-commerce-Dataset-For-D2T
    
The code uses the library click which allows to use python scripts like linux commands with options. The descriptions of how to use the scripts are contained in the scripts and can be viewed by using the "--help" option with the corresponding script.

To prepare the datasets run prepare_data.py from the main directory.
To list the options and their descriptions run:

    python src/prepare_data.py --help

To train the model run train.py from the main directory.
To list the options and their descriptions (including the values used in the thesis) run:

    python src/train.py --help

To predict on a test set or to generate a sentence out of a data input run predict.py from the main directory with the corresponding options.
To list the options and their descriptions run:

    python src/predict.py --help

To automatically evaluate the predictions first clone the GenerationEval repository:

    git clone https://github.com/WebNLG/GenerationEval.git

Then download the bleurt-base-128 checkpoint (https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip) and place it in "GenerationEval/metrics/bleurt/"

Follow the instructions in the repo and use the references created by prepare_data.py to evaluate the predictions.

The code that was used for the manual evaluations and the human evaluation is also included in the directory and is also explained when using the "--help" option. As the data is already provided 

To get statistics of a dataset and recreate the statistical comparison of datasets in the masters thesis run the data_analysis.py script.
To list the options and their descriptions run:

    python src/data_analysis.py --help

As the surveys used in thesis are already provided you do not have to use the scripts get_test_lengths.py, first_manual_survey.py, second_manual_survey.py and human_survey.py.

To get the results of the first and second manual evaluation run the eval_manual_surveys.py script.
To list the options and their descriptions run:

    python src/eval_manual_surveys.py --help

To get the results of the human evaluation run the human_eval.py script.
To list the options and their descriptions run:

    python src/human_eval.py --help
