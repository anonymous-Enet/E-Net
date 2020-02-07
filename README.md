Robust Network Enhancement from Flawed Networks
===============================================================================

About
-----

This project implements the E-net model, which focuses on reconstructing a reliable network from a flawed (noisy) network.

Dependencies
-----

The script has been tested running under Python 2.7.12, with the following packages installed (along with their dependencies):

- `torch==1.1.0`
- `networkx==2.0`
- `sklearn==0.19.1`
- `numpy==1.11.0`
- `scipy==1.1.0`
- `gensim==3.6.0`
- `tqdm==4.19.4`

In addition, CUDA 8.0 has been used.

Input Format
-----
An example data format is given in ```Enet/data``` where dataset is in ```*.mat``` format and ```*.score``` is some heuristic scores calculated in advance.

How to run E-net
-----
    
    cd Enet
    
    python Main.py -h
    
    usage: Main.py [-h][--data-name] [--use-embedding ] [--num-walks] [--learning-rate] [--reg-smooth] [--smooth-coef] [--trainable-noise]
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset             str, select the dataset. 
      --use-embedding       bool, whether to use node2vec node embeddings as additional information.
      --num-walks           int, number of walks for each node when performing Lazy Subgraph Extraction. 
      --learning-rate	    float, the learning rate. 
      --reg-smooth          bool, whether to use auxiliary denoising regularization.
      --smooth-coef         float, the coefficient of auxiliary denoising regularization. 
      --trainable-noise     bool, whether to let the Noisy link detection layer trainable.

To reproduce the results that reported in the paper, you can run the following command:

    python Main.py --data-name citeseer --use-embedding --num-walks 5 --learning-rate 1e-4 --noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
