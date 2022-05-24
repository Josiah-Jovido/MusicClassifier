import random

nf = ["play 1", "play 2", "play 3", "play 4", "play 5"]

def next(x):
    """next button"""
    if x >= 1:
        random.sample(nf, 2)
    else:
        pass
 