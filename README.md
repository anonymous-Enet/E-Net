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

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 8.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Input Format
-----
An example data format is given in ```Enet/data``` where dataset is in ```*.mat``` format and ```*.score``` is some heuristic scores calculated in advance.

How to run E-net
-----
    
    cd Enet
    
    python Main.py -h
    
    usage: Main.py [-h][--data-name] [--save-name] [--max-train-num] [--no-cuda] [--missing-ratio] 
    [--split-ratio] [--neg-pos-ratio] [--use-attribute] [--use-embedding] [--embedding-size] 
    [--lazy-subgraph] [--max-nodes-per-hop] [--num-walks] [--multi-subgraph] [--reg-smooth] 
    [--smooth-coef] [--trainable-noise] [--early-stop] [--early-stop-patience] [--learning-rate] 
    
    optional arguments:
      -h, --help                show this help message and exit
      --data-name               str, select the dataset. 
      --save-name               str, the name of saved model. 
      --max-train-num           int, the maximum number of training links.
      --no-cuda                 bool, whether to disables CUDA training.
      --seed                    int, set the random seed.
      --test-ratio              float, the ratio of test links.
      --missing-ratio           float, the ratio of missing links.
      --split-ratio             str, the split rate of train, val and test links
      --neg-pos-ratio           float, the ratio of negative/positive links
      --use-attribute           bool, whether to utilize node attribute. 
      --use-embedding           bool, whether to utilize the information from node2vec node embeddings.
      --embedding-size          int, the embedding size of node2vec
      --lazy-subgraph           bool, whether to use lazy subgraph extraction.
      --max-nodes-per-hop       int, the upper bound the number of nodes per hop when performing Lazy Subgraph Extraction. 
      --num-walks               int, thenumber of walks for each node when performing Lazy Subgraph Extraction. 
      --multi-subgraph          int, the number of subgraphs to extract for each queried nodes
      --reg-smooth              bool, whether to use auxiliary denoising regularization.
      --smooth-coef             float, the coefficient of auxiliary denoising regularization. 
      --trainable-noise         bool, whether to let the Noisy link detection layer trainable.
      --early-stop              bool, whether to use early stopping.
      --early-stop-patience     int, the patience for early stop.
      --learning-rate           float, the learning rate. 

To reproduce the results that reported in the paper, you can run the following command:

    python Main.py --data-name citeseer --use-embedding --num-walks 5 --learning-rate 1e-4 
    --noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
