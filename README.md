Robust Network Enhancement from Flawed Networks
===============================================================================

About
-----

Code for E-net. 

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

How to run E-net
-----

For PPDai dataset:

    python Main.py --data-name ppd_rule_new --num-walks 3 --learning-rate 1e-3 --noise-hidden-dim 300 --reg-smooth --smooth-coef 1e-4 --trainable-noise --save-name _ours --use-sig --noise-init
    
For Cora dataset:

    python Main.py --data-name cora --use-embedding --num-walks 30 --learning-rate 1e-4 --smooth-coef 1e-4 --noise-hidden-dim 300 --trainable-noise --save-name _ours --use-soft --use-sig --num-node-to-walks 2 --reg-smooth --smooth-coef 1e-4
    
For Citeseer dataset:

    python Main.py --data-name citeseer --use-embedding --num-walks 5 --learning-rate 1e-4 --noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
       
For Pubmed dataset:

     python Main.py --data-name pubmed --num-walks 10 --learning-rate 1e-4 --noise-hidden-dim 500 --reg-smooth --smooth-coef 1e-3 --trainable-noise --save-name _best_ours --use-sig --use-soft # E-Net
