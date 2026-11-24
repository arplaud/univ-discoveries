import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['figure.dpi']=1000

ax1 = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])

# Panel (a) - Heterogeneous competitors

list_frac_trans= [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.5 hetero') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.5 hetero/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_trans[i] = np.concatenate((list_frac_trans[i],f[i]))    
          
list_var_trans = [np.var(list_frac_trans[i]) for i in range(50)]

list_frac_inter = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.75 hetero') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.75 hetero/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_inter[i] = np.concatenate((list_frac_inter[i],f[i]))

list_var_inter= [np.var(list_frac_inter[i]) for i in range(50)]

list_frac_recu = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy 1D alpha = 1.5 hetero') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy 1D alpha = 1.5 hetero/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_recu[i] = np.concatenate((list_frac_recu[i],f[i]))

list_var_recu= [np.var(list_frac_recu[i]) for i in range(50)]

n = np.logspace(0,7,50)
n_rec=np.logspace(1,5,50)

ax1.scatter(n_rec[3:],list_var_recu[3:],color='b',s=3,label=r'$d_s=4/3$')
ax1.plot(n_rec[22:],0.012*n_rec[22:]**0,color='b',lw=1)
ax1.annotate(r'$\sim \mathrm{Var}[X_\infty]$',(4e3,3e-3),color='b',fontsize=16)

ax1.scatter(n[9:],list_var_inter[9:],color='r',s=3,label=r'$d_s=8/3$')
ax1.plot(n[22:], 0.11/n[22:]**(2/3), color='r', lw=1)
ax1.annotate(r'$\sim n^{-2/3}$', (4*10**5, 3e-5), color='r', fontsize=16)

ax1.scatter(n[9:],list_var_trans[9:],color='g',s=3,label=r'$d_s=4$')
ax1.plot(n[22:], 0.12/n[22:]**(1), color='g', lw=1)
ax1.annotate(r'$\sim n^{-1}$', (4*10**4, 1e-7), color='g', fontsize=16)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'number of discovered sites $n$', fontsize=12)
ax1.set_ylabel(r'$\mathrm{Var}(X_n)$', fontsize=12)
ax1.yaxis.set_label_coords(-0.12, 0.5)
ax1.set_xlim(10, 2*10**7)
ax1.set_ylim(1e-8, 2e-2)

img = mpimg.imread('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_hetero.png')
imagebox= OffsetImage(img, zoom=0.036)
a = AnnotationBbox(imagebox,(5e2,2e-6), frameon=False)
ax1.add_artist(a)

ax1.legend(framealpha=1, facecolor='white', markerscale=2, fontsize=12,loc='upper right')
ax1.set_title(r'\textbf{Unequal mobilities}', fontsize=15,y=1.05)

ax1.grid(False)

# Panel (b) - Arbitrary number of competitors

list_frac_inter = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.75 n = 4') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 0.75 n = 4/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_inter[i] = np.concatenate((list_frac_inter[i],f[i]))

list_var_inter= [np.var(list_frac_inter[i]) for i in range(50)]

list_frac_recu = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 1.5 n = 7') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 1.5 n = 7/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_recu[i] = np.concatenate((list_frac_recu[i],f[i]))
            
list_var_recu = [np.var(list_frac_recu[i]) for i in range(50)]

list_frac_trans = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha=0.5 n=5') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha=0.5 n=5/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_trans[i] = np.concatenate((list_frac_trans[i],f[i]))
            
list_var_trans = [np.var(list_frac_trans[i]) for i in range(50)]

n = np.logspace(0,6,50)
n_trans = np.logspace(1,7,50)

ax2.scatter(n[11:36],list_var_recu[11:36],color='b',s=3,label=r'$d_s=4/3$')
ax2.plot(n[22:36], 0.01/n[22:36]**(0), color='b', lw=1)
ax2.annotate(r'$\sim \mathrm{Var}[X_\infty]$', (10**3, 2e-3), color='b', fontsize=16)

ax2.scatter(n[11:],list_var_inter[11:],color='r',s=3,label=r'$d_s=8/3$')
ax2.plot(n[22:], 0.065/n[22:]**(2/3), color='r', lw=1)
ax2.annotate(r'$\sim n^{-2/3}$', (1.2*10**5, 5e-5), color='r', fontsize=16)

ax2.scatter(n_trans[3:42],list_var_trans[3:42],color='g',s=3,label=r'$d_s=4$')
ax2.plot(n[22:],0.08/n[22:]**1,color='g',lw=1)
ax2.annotate(r'$\sim n^{-1}$',(2e4,2e-7),color='g',fontsize=16)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'number of discovered sites $n$', fontsize=12)
ax2.set_ylabel(r'$\mathrm{Var}(X_n)$', fontsize=12)
ax2.yaxis.set_label_coords(-0.12, 0.5)
ax2.set_xlim(10, 1.5*10**6)
ax2.set_ylim(3e-8, 2e-1)

img = mpimg.imread('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_many.png')
imagebox= OffsetImage(img, zoom=0.024)
a = AnnotationBbox(imagebox,(0.8e2,2e-6), frameon=False)
ax2.add_artist(a)

ax2.legend(framealpha=1, facecolor='white', markerscale=2, fontsize=12,loc='upper right')
ax2.set_title(r'\textbf{Multiple competitors}', fontsize=15,y=1.05)

ax2.grid(False)

# Panel (c) - Walks on fractal substrates

list_frac_perc = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction percolation cluster N = 10**1 to 10**4') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction percolation cluster N = 10**1 to 10**4/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_perc[i] = np.concatenate((list_frac_perc[i],f[i]))

list_var_perc = [np.var(list_frac_perc[i]) for i in range(50)]

list_frac_sierpinski = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction sierpinski N= 10**1 to 10**5') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction sierpinski N= 10**1 to 10**5/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_sierpinski[i] = np.concatenate((list_frac_sierpinski[i],f[i]))

list_var_sierpinski = [np.var(list_frac_sierpinski[i]) for i in range(50)]

n = np.logspace(1,4,50)

ax3.scatter(n[2:],list_var_perc[2:],color='purple',s=3,label = r'percolation cluster, $d_s \approx 1.317$')
ax3.plot(n[22:], 0.027/n[22:]**(0), color='purple', lw=1)
ax3.annotate(r'$\sim \mathrm{Var}[X_\infty]$', (3*10**3, 1.5e-2), color='purple', fontsize=16)

n = np.logspace(1,5,50)

ax3.scatter(n[2:],list_var_sierpinski[2:],color='b',s=3, label = r'Sierpinski triangle, $d_s \approx 1.365$')
ax3.plot(n[22:], 0.035/n[22:]**(0), color='b', lw=1)
ax3.annotate(r'$\sim \mathrm{Var}[X_\infty]$', (3*10**3, 6e-2), color='b', fontsize=16)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'number of discovered sites $n$', fontsize=12)
ax3.set_ylabel(r'$\mathrm{Var}(X_n)$', fontsize=12)
ax3.yaxis.set_label_coords(-0.12, 0.5)
ax3.set_yticks([1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1,2e-1,3e-1,4e-1],[r'$10^{-3}$',None,None,None,None,None,None,None,None,r'$10^{-2}$',None,None,None,None,None,None,None,None,r'$10^{-1}$',None,None,None],minor = True)
ax3.set_xlim(10, 1.5*10**5)
ax3.set_ylim(1e-3,4e-1)

img = mpimg.imread('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_fractals.png')
imagebox= OffsetImage(img, zoom=0.016)
a = AnnotationBbox(imagebox,(2e2,5e-3), frameon=False)
ax3.add_artist(a)

ax3.legend(framealpha=1, facecolor='white', markerscale=2, fontsize=12,loc='upper right')
ax3.set_title(r'\textbf{Heterogeneous substrates}', fontsize=15,y=1.05)

ax3.grid(False)

# Panel (d) - 2D Fractional Brownian Motion

list_frac_recu = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction FBM 2D H=0.33 N = 10**1 to 10**4') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction FBM 2D H=0.33 N = 10**1 to 10**4/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_recu[i] = np.concatenate((list_frac_recu[i],f[i]))

list_var_recu = [np.var(list_frac_recu[i]) for i in range(50)]

list_frac_inter = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction fbm H=0.66 N = 10 to 10**6') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction fbm H=0.66 N = 10 to 10**6/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_inter[i] = np.concatenate((list_frac_inter[i],f[i]))

list_var_inter = [np.var(list_frac_inter[i]) for i in range(50)]

list_frac_trans = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction fbm 3D H=0.66 N=10 to 10**6') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction fbm 3D H=0.66 N=10 to 10**6/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_trans[i] = np.concatenate((list_frac_trans[i],f[i]))

list_var_trans = [np.var(list_frac_trans[i]) for i in range(50)]

n = np.logspace(1,4,50)
ax4.scatter(n[2:],list_var_recu[2:],color='b',s=3,label = r'$d_s = 4/3$')
ax4.plot(n[20:],0.0095/n[20:]**0,color='b',lw=1)
ax4.annotate(r'$\sim \mathrm{Var}[X_\infty]$',(1e3,2.8e-3),color='b',fontsize=16)

n=np.logspace(1,6,50)
ax4.scatter(n[2:],list_var_inter[2:],color='r',s=3,label = r'$d_s = 8/3$')
ax4.plot(n[20:],0.055/n[20:]**(2/3),color='r',lw=1)
ax4.annotate(r'$\sim n^{-2/3}$',(1.2e5,4e-5),color='r',fontsize=16)

ax4.scatter(n[2:],list_var_trans[2:],color='g',s=3,label = r'$d_s = 4$')
ax4.plot(n[20:],0.025/n[20:],color='g',lw=1)
ax4.annotate(r'$\sim n^{-1}$',(1.8e4,0.8e-7),color='g',fontsize=16)

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel(r'number of discovered sites $n$', fontsize=12)
ax4.set_ylabel(r'$\mathrm{Var}(X_n)$', fontsize=12)
ax4.yaxis.set_label_coords(-0.12, 0.5)
ax4.set_xlim(10, 1.6*10**6)
ax4.set_ylim(1e-8,3e-2)

img = mpimg.imread('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_fBm.png')
imagebox= OffsetImage(img, zoom=0.033)
a = AnnotationBbox(imagebox,(2e2,3e-6), frameon=False)
ax4.add_artist(a)

ax4.legend(framealpha=1, facecolor='white', markerscale=2, fontsize=12,loc='upper right')
ax4.set_title(r'\textbf{Memory-driven transport}', fontsize=15,y=1.05)

ax4.grid(False)

ytop = (ax1.title.get_window_extent().y0 + ax1.title.get_window_extent().y1)/16000
ymiddle = (ax4.title.get_window_extent().y0 + ax4.title.get_window_extent().y1)/16000


fig.text(0.09,ytop,r'\textbf{(a)}',fontsize=15,va='center')
fig.text(0.09,ymiddle,r'\textbf{(c)}',fontsize=15,va='center')
fig.text(0.52,ytop,r'\textbf{(b)}',fontsize=15,va='center')
fig.text(0.52,ymiddle,r'\textbf{(d)}',fontsize=15,va='center')

#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.patches as patches

# -----------------------------
# Paramètres
# -----------------------------
L = 20
p = 0.5
seed = 1234

rng = np.random.default_rng(seed)

# Liens ouverts
horizontal = rng.random((L, L - 1)) < p
vertical   = rng.random((L - 1, L)) < p

# -----------------------------
# Voisins d'un site
# -----------------------------
def neighbors(x, y):
    if x > 0 and horizontal[y, x - 1]:
        yield (x - 1, y)

    if x < L - 1 and horizontal[y, x]:
        yield (x + 1, y)

    if y > 0 and vertical[y - 1, x]:
        yield (x, y - 1)

    if y < L - 1 and vertical[y, x]:
        yield (x, y + 1)

# -----------------------------
# Recherche des amas
# -----------------------------
visited = np.zeros((L, L), dtype=bool)

largest_cluster = []

for x0 in range(L):
    for y0 in range(L):

        if visited[y0, x0]:
            continue

        cluster = []
        queue = deque([(x0, y0)])
        visited[y0, x0] = True

        while queue:

            x, y = queue.popleft()
            cluster.append((x, y))

            for xn, yn in neighbors(x, y):

                if not visited[yn, xn]:
                    visited[yn, xn] = True
                    queue.append((xn, yn))

        if len(cluster) > len(largest_cluster):
            largest_cluster = cluster

# -----------------------------
# Affichage du plus grand amas
# -----------------------------
cluster_set = set(largest_cluster)

fig, ax = plt.subplots(figsize=(8, 8))
plt.rcParams['figure.dpi']=1000

# liens internes à l'amas
for x, y in largest_cluster:

    if x < L - 1 and horizontal[y, x]:
        if (x + 1, y) in cluster_set:
            ax.plot([x, x + 1], [y, y],
                    color='black', lw=4)

    if y < L - 1 and vertical[y, x]:
        if (x, y + 1) in cluster_set:
            ax.plot([x, x], [y, y + 1],
                    color='black', lw=4)

# sites
X = [x for x, y in largest_cluster]
Y = [y for x, y in largest_cluster]

ax.scatter(X, Y, s=80, color='crimson',zorder=20)
'''
ax.scatter([8,9],[11,11],s=200,color='red',zorder=50)

style2 = "Simple, tail_width=3, head_width=30, head_length=25"

kw3 = dict(arrowstyle=style2,color='green',zorder=200)

ax.add_patch(patches.FancyArrowPatch((8,11),(9,11),connectionstyle="arc3,rad=10",**kw3))
'''
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()

fig.savefig('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_fractals.png',transparent=True)

#%%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage as ndimage

def Discrete2D(t):
    x,y = [0],[0]
    for i in range(t):
        t = np.random.random()
        if t<0.25:
            x.append(-1)
            y.append(0)
        elif t < 0.5:
            x.append(1)
            y.append(0)
        elif t<0.75:
            x.append(0)
            y.append(-1)
        else:
            x.append(0)
            y.append(1)
    listx,listy = np.cumsum(x),np.cumsum(y)
    return(listx,listy)

t=2000
##seed = np.random.randint(10**6)
##print(seed)
np.random.seed(723743)

x1,y1 = Discrete2D(t)
x2,y2 = Discrete2D(t)
x3,y3 = Discrete2D(t)
x4,y4 = Discrete2D(t)

sites_visited = set()
discoveries_1,discoveries_2,discoveries_3,discoveries_4 = [],[],[],[]
for i in range(t):
    newpos1 = x1[i],y1[i]
    if newpos1 not in sites_visited:
        sites_visited.add(newpos1)
        discoveries_1.append(newpos1)
    newpos2 = x2[i],y2[i]
    if newpos2 not in sites_visited:
        sites_visited.add(newpos2)
        discoveries_2.append(newpos2)
    newpos3 = x3[i],y3[i]
    if newpos3 not in sites_visited:
        sites_visited.add(newpos3)
        discoveries_3.append(newpos3)
    newpos4 = x4[i],y4[i]
    if newpos4 not in sites_visited:
        sites_visited.add(newpos4)
        discoveries_4.append(newpos4)

from matplotlib.colors import ListedColormap

all_cells = discoveries_1+discoveries_2+discoveries_3+discoveries_4

R1,R2,R3,R4 = int(1.1*max(x for x, y in all_cells)),int(1.1*max(-x for x, y in all_cells)),int(1.1*max(y for x, y in all_cells)),int(1.1*max(-y for x, y in all_cells))

size = max(R1+R2+1,R3+R4+1)


# --- Paramètres ---
offset_x = R2+4
offset_y = R3-1
# pour recentrer les coordonnées

grid = np.zeros((size, size))

# --- Remplissage ---
for x, y in discoveries_1:
    i = offset_y - y
    j = offset_x + x
    if 0 <= i < size and 0 <= j < size:
        grid[i, j] = 1  # rouge

for x, y in discoveries_2:
    i = offset_y - y
    j = offset_x + x
    if 0 <= i < size and 0 <= j < size:
        grid[i, j] = 2  # bleu 

for x, y in discoveries_3:
    i = offset_y - y
    j = offset_x + x
    if 0 <= i < size and 0 <= j < size:
        grid[i, j] = 3  # vert

for x, y in discoveries_4:
    i = offset_y - y
    j = offset_x + x
    if 0 <= i < size and 0 <= j < size:
        grid[i, j] = 4  # orange

grid[R3-1,R2+4] = 5

# --- Colormap ---
#cmap = ListedColormap(["#D3D3D3", "#B83232", "#346B96","#29271B"])
cmap = ListedColormap(["#FFFFFF", "#B52B2B", "#316690","#46825A","#D28C3C","#29271B"])

import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

plt.rcParams['figure.dpi']=1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1,1)

ax = fig.add_subplot(gs[0,0])

ax.imshow(grid, cmap=cmap)

'''
ax.axhline(y=offset, color='black', linewidth=20/R)  # axe horizontal (y=0)
ax.axvline(x=offset, color='black', linewidth=20/R)
'''

# Grille
ax.set_xticks(np.arange(-.5, size, 1))
ax.set_yticks(np.arange(-.5, size, 1))
ax.tick_params(
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False
)
ax.grid(False)
ax.axis('off')

# Enlever les labels
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_aspect('equal')

fig.savefig('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_many.png',transparent=True)

#%%

plt.rcParams['figure.dpi']=1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1,1)

ax = fig.add_subplot(gs[0,0])

ax.scatter(-2,0,s=150,color='purple',zorder=50)
ax.scatter(3,0,s=600,color='orange',zorder=50)

ax.axis('off')
ax.hlines(0,-4.5,6.5,lw=2,color='black')
ax.vlines([-4,-3,-2,-1,0,1,2,3,4,5,6],0,0.02,lw=2,color='black')
ax.set_ylim(-0.2,1)

style2 = "Simple, tail_width=1, head_width=9, head_length=5"

kw3 = dict(arrowstyle=style2,color='orange',zorder=20)
ax.add_patch(patches.FancyArrowPatch((3,0),(5,0),connectionstyle="arc3,rad=-4",**kw3))

kw3 = dict(arrowstyle=style2,color='purple',zorder=20)
ax.add_patch(patches.FancyArrowPatch((-2,0),(-1,0),connectionstyle="arc3,rad=-6",**kw3))

fig.savefig('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_hetero.png',transparent=True)

#%%

plt.rcParams['figure.dpi']=1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.patches as patches

def generate_gaussian(N):
    Vj = np.zeros((2 * N, 2), dtype=np.complex128)
    Vj[0, 0] = np.random.standard_normal()
    Vj[N, 0] = np.random.standard_normal()
    for i in range(1, N):
        Vj1 = np.random.standard_normal()
        Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1
        Vj[i][1] = Vj2
        Vj[2 * N - i][0] = Vj1
        Vj[2 * N - i][1] = Vj2
    return Vj


# Transforms the gaussian sample into fbm sample
def davies_harte(T, N, H, Vj):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method from the list of Gaussian rv

    Args:
        T (float): Length of time
        N (int): Number of time steps within timeframe
        H (float): Hurst parameter
        Vj (2N, 2 array): periodic matrix of Gaussian rv

    Returns:
        numpy.ndarray: Generated path of fractional Brownian Motion
    '''
    # Step 0: Initialize parameters
    def gamma(k, H): return 0.5 * (np.abs(k - 1) ** (2 * H) -
                                   2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    g = [gamma(k, H) for k in range(0, N)]
    r = g + [0] + g[::-1][0:N - 1]

    # Step 1: Compute eigenvalues
    j = np.arange(0, 2 * N)
    k = 2 * N - 1
    lk = np.fft.fft(r * np.exp(2 * np.pi * complex(0, 1)
                    * k * j * (1 / (2 * N))))[::-1]

    # Step 2: Compute Z
    wk = np.zeros(2 * N, dtype=np.complex128)
    wk[0] = np.sqrt((lk[0] / (2 * N))) * Vj[0][0]
    wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * \
        ((Vj[1:N].T[0]) + (complex(0, 1) * Vj[1:N].T[1]))
    wk[N] = np.sqrt((lk[0] / (2 * N))) * Vj[N][0]
    wk[N + 1:2 * N] = np.sqrt(lk[N + 1:2 * N] / (4 * N)) * (
        np.flip(Vj[1:N].T[0]) - (complex(0, 1) * np.flip(Vj[1:N].T[1])))

    Z = np.fft.fft(wk)
    fGn = np.real(Z[0:N])
    fBm = np.cumsum(fGn) * (N ** (-H))
    fBm = (T ** H) * fBm
    path = np.array([0] + list(fBm))

    return path

seed = np.random.randint(10000)
np.random.seed(6707)

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1,1)

ax = fig.add_subplot(gs[0,0])

ax.plot(davies_harte(10,30,0.9,generate_gaussian(30)),davies_harte(10,30,0.9,generate_gaussian(30)),lw=4,color='purple')
ax.plot(davies_harte(10,30,0.3,generate_gaussian(30)),davies_harte(10,30,0.3,generate_gaussian(30)),lw=4,color='orange')

ax.axis('off')

ax.annotate(r'$H>1/2$',(-10,6),color='purple',fontsize=25)
ax.annotate(r'$H<1/2$',(-4,-2),color='orange',fontsize=25)

fig.savefig('/users/invites/plaud/Documents/articles/article competitive foraging/illu_robustness_fBm.png',transparent=True)







