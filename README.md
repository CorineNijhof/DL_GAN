# DL_GAN
A project for the University of Groningen course Deep Learning.

## To run the program:
Download the dataset (https://www.kaggle.com/ipythonx/van-gogh-paintings)  
extract dataset and rename to 'data'
for running on a subset of this dataset, put the relevant directories into a separate directory and give the name of this directory as the dataset parameter
for running on the 'paris' dataset, put the directory Paris into a separate directory called 'paris'

The command for running the code is
python3 launch.py

Additional parameters can be added in the following fixed order (first option is default):
1. choose the dataset (data, choose your own directory containing a subset of the dataset, option 'test' tests if the networks are compatible)
2. choose the network (default, default128, VANGAN)
3. choose the learning rate (0.0002, any value possible)
4. choose the optimizer for the discriminator (Adam, SGD)
5. choose the number of epochs (default: 250, VANGAN: 550, any value possible)

e.g. python3 launch.py data default 0.0002 Adam 250
