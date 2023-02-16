# Data2Text - Model for WebNLG 2020+ data set and AX generated data set

To create the data files needed for the WebNLG data set, first you have to clone the following git repository into the main directory:

    git clone https://gitlab.com/shimorina/webnlg-dataset.git

Then run the prepare_data.py script from the main directory:

    python src/prepare_data.py

To train the model run train.py from the main directory.
To list the options and see the default values run:

    python src/train.py --help

To predict predict on a test set or to generate a sentence out of a data input run predict.py from the main directory with the corresponding options.
To list the options and see the default values run:

    python src/predict.py --help

To evaluate the predictions first clone the GenerationEval repository:

    git clone https://github.com/WebNLG/GenerationEval.git

Then download the bleurt-base-128 checkpoint (https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip) and place it in "GenerationEval/metrics/bleurt/"

Follow the instructions in the repo and use the references created by prepare_data.py to evaluate the predictions.
