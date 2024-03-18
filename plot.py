import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os,sys  

try:
    os.mkdir('result')
except:
    pass

data = []
with open("validation_loss.dat","r") as f:
    data = f.readlines()

d = {
    0: "CPU_BLAS",
    1: "GPU_CuBLAS",
    2: "CPU_Native",
    3: "GPU_Native"
}

data=[i.strip().replace('\n','').strip().split(" ") for i in data]
nl = int(data[0][0])
nh = int(data[0][1])
ne = int(data[0][2])
nb = int(data[0][3])
alpha = float(data[0][4])
type = d[int(data[0][5])]

data.pop(0)
data = [[float(i) for i in j][0] for j in data]

plt.plot(list(range(1,len(data)+1)), data, label='Validation Loss', color='royalblue', linestyle='-')
plt.title(f"{type}\nlayers = {[784]+[nh]*nl+[10]}, batch_size = {nb}, epoch = {ne}, LR = {alpha}", fontsize=12, fontweight='normal')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

filename = f"result/{type}_validation_plot.png"
plt.savefig(filename,dpi=300)

print(filename)