import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns

# Function to read and parse data from the file
def read_diamond_data(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

        iline = 0
        while iline < len(lines):
            line = lines[iline]
            d = {}

            l1 = line.strip().split()[1:]
            l2 = lines[iline + 1].strip().split()[1:]

            for k, v in zip(l1, l2):
                if k[-1] == ",":
                    k = k[:-1]
                
                if v[-1] == ",":
                    v = v[:-1]

                d[k] = float(v)

            data.append(d)
            iline += 3

    return data

c0_list = set()
rela_qr_list = set()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--fname', type=str, default='diamond')
args = parser.parse_args()
fname = args.fname

data = read_diamond_data(f'{args.fname}.out')
for d in data:
    print(d)
    c0_list.add(d['c0'])
    rela_qr_list.add(d['rela_qr'])

fig, ax = plt.subplots(figsize=(6, 3))
import itertools

# sort by c0, rela_qr
xmin = None
xmax = None
c0_rela_qr_list = sorted(list(itertools.product(c0_list, rela_qr_list)), key=lambda x: (x[0], x[1]))
for c0, rela_qr in c0_rela_qr_list:
    if rela_qr > 1e-3:
        continue
    
    kmin = 0.0
    if fname == 'diamond':
        kmin = 50.0

    if fname == 'nio':
        kmin = 100.0

    x = [d['ke_cutoff'] for d in data if d['c0'] == c0 and d['rela_qr'] == rela_qr]
    y = [d['e_tot'] for d in data if d['c0'] == c0 and d['rela_qr'] == rela_qr]
    perm = np.argsort(x)
    x = np.array(x)[perm]
    y = np.array(y)[perm]
    m = x >= kmin
    x = x[m]
    y = y[m]
    ax.plot(x, y, label=f'c0 = {c0}, rela_qr = {rela_qr}')

    if xmin is None:
        xmin = x[0]
    else:
        xmin = min(xmin, x[0])

    if xmax is None:
        xmax = x[-1]
    else:
        xmax = max(xmax, x[-1])

ax.set_xlim(xmin, xmax)

ax.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)

fig.savefig(f'{fname}.png', bbox_inches='tight')