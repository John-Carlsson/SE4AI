# SE4AI
This is repository for the project within the course Software Engineering for Artificial intelligence. This group consist of 7 Erasmus students at the University of Salerno

## Getting Started

### Setting up environment
First, we recommend to create an environment to install the appropriate Python packages. For this we recommend to use conda (https://conda.io/projects/conda/en/latest/index.html). 

After installing Anaconda on your system (https://www.anaconda.com/download/), you can create a conda environment. Inside the repository is a environment.yml file which contains all necessary packages to execute the code. The environment including the packages can be created and installed by executing the following command: 
console
conda env create -f environment.yml
 

The name of created environment <env_name> should be env_gen_ai_cs595. This environment can then be activated using the following command. 
console
conda activate <env_name>
 

Depending on the IDE used, further steps may be required to add the environment as a kernel. In Jupyter it might be needed to add your environment to Jupyter with:
console
python -m ipykernel install --user --name=<env_name>

### Starting the Application

To start the combined Application just run main.py, to start the FER application run gui_fer.py and to start the SER application run the file main_ser.py in the SER folder.


## Datasets

Dataset FER: https://www.kaggle.com/datasets/deadskull7/fer2013
Dataset SET: https://www.kaggle.com/datasets/ejlok1/cremad

  
## Models

For FER we have the class NNmodel that contains some fundamental functions for training NN for facial emotion detection. 
Sofar only one (sequential) model is implemented, which provides a 66% accuracy on the validation set. This is 
quite good, since our expectation is an accuracy of 70%. 

train_model() saves the trained model and returns the path of the trained model.
[Note: training this model takes approx. 30 min.]
  
For SER the file model_ser.py contains the models structure.

