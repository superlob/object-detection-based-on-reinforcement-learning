from torchvision import transforms
import torch.nn as nn
import torch
from env import env
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
path1 = 'test_exp'


s = env.reset()
env.current_box = [20,110,280,330]
#env.draw_block(0,21)
print(env.target_box)
env.step(1,0)
env.draw_block(0,21)