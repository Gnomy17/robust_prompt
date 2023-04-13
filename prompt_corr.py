from utils import *
import torch
from model_for_cifar import prompt
from model_for_cifar.vit import vit_base_patch16_224
import numpy as np
import matplotlib.pyplot as plt

chkpnt = torch.load(r'saved_chkpnts/natural_pr/natural_pr_epch4')
p1 = chkpnt['prompt']['prompt'].cpu().numpy().squeeze()
chkpnt = torch.load(r'saved_chkpnts/natural_pr/natural_pr_epch4')
p2 = chkpnt['prompt']['prompt'].cpu().numpy().squeeze()
        
corr_mat = np.zeros((p1.shape[0], p2.shape[0]))
def cos_sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
for i in range(p1.shape[0]):
    for j in range(p2.shape[0]):
        corr_mat[i,j] = cos_sim(p1[i,:], p2[j,:])
plt.matshow(corr_mat)
plt.savefig("figs/correlation_mats/nat_itself.png", dpi=500)    
    