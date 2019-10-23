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

    python Main.py --data-name citeseer --use-embedding --num-walks 5 --learning-rate 1e-4 --noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
