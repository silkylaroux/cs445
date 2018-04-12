import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as pltpatch  # for Arc
# import matplotlib.collections as pltcoll
import pandas as pd
import sys
import random
import os


if __name__ == "__main__":
    height,width = 10,13
    
    basic_map = np.zeros(shape =(height, width))
    basic_map[:] = 0
    print(basic_map)
    os.system('cls')
    basic_map[3][7] = 5
    print(basic_map)

    