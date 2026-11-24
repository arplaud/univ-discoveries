import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.optimize import curve_fit
from scipy.stats import kurtosis
from scipy.special import beta


list_frac_two_sided = []

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D/' + file, allow_pickle = True)
        list_frac_two_sided = np.concatenate((list_frac_two_sided,f))

list_frac_one_sided = []

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D one sided') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D one sided/' + file, allow_pickle = True)
        list_frac_one_sided = np.concatenate((list_frac_one_sided,f))




def f(x,a,b):
    return(a*x**(-2/3) - b*x**(-1))

x = np.linspace(0,1,200)


import matplotlib.gridspec as gridspec

np.random.seed(294298)

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], hspace=0.25, wspace=0.35)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['figure.dpi']=1000

ax2 = fig.add_subplot(gs[1, 0])
ax = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[0, :])

# -----------------------------
# Panel (b) : Symmetric distribution
# -----------------------------
ax.hist(list_frac_two_sided, color='b', density = True,bins = 200,histtype = 'stepfilled',cumulative = False,label=r'Simulated distribution of $X_\infty$',alpha=0.6)
##ax.plot(x,(x*(1-x))**2.14/beta(3.14,3.14),color='r',lw=0.3,label=r'arcsine limiting law')

ax.set_xlabel(r'value of the discovery share $x$', fontsize=13)
ax.set_ylabel(r'probability density $\rho(x)$', fontsize=13)
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.set_xlim(0,1)
ax.set_ylim(0,2)
ax.legend(framealpha=1,loc= 'lower center', facecolor='white', markerscale=10, fontsize=12)

ax.set_title(r'\textbf{Discovery share distribution in the two-sided case}', fontsize=15,y=1.05)

ax.grid(False)

# -----------------------------
# Panel (c) : One-sided distribution
# -----------------------------

ax2.hist(list_frac_one_sided, color='b', density = True,bins = 200,histtype = 'stepfilled',cumulative = False,label=r'Simulated distribution of $X_\infty$',alpha=0.6)
ax2.plot(x,1/(np.pi*np.sqrt(x*(1-x))),color='r',lw=1.3,label=r'arcsine limiting law')

ax2.set_xlabel(r'value of the discovery share $x$', fontsize=13)
ax2.set_ylabel(r'probability density $\rho(x)$', fontsize=13)
ax2.yaxis.set_label_coords(-0.12, 0.5)
ax2.set_xlim(0,1)
ax2.set_ylim(0,4)
ax2.legend(framealpha=1, loc = 'upper center', facecolor='white', markerscale=10, fontsize=12)

ax2.set_title(r'\textbf{Discovery share distribution in the one-sided case}', fontsize=15,y=1.05)

ax2.grid(False)

# ------------------------------
# Top Panel : Illustrations
# ------------------------------

style1 = "Simple, tail_width=0.1, head_width=8, head_length=8"
kw1 = dict(arrowstyle=style1, color="k")

style1_double = "<|-|>, head_width=4, head_length=8"
kw1_double = dict(arrowstyle=style1_double, color="k")

style2 = "Simple, tail_width=0.5, head_width=3, head_length=3"
kw2 = dict(arrowstyle=style2,color = 'b')

kw3 = dict(arrowstyle=style2,color='green')

ax3.set_xlim(0,2)
ax3.set_ylim(-1.6,2)
delta=0.4
c1=(-ax2.get_position().x0 + ax2.get_position().x1)/(ax.get_position().x1 - ax2.get_position().x0)
c2=2 - (-ax.get_position().x0 + ax.get_position().x1)/(ax.get_position().x1 - ax2.get_position().x0)
n=8

ax3.add_patch(patches.FancyArrowPatch((c1-delta,0),(c1+delta,0),**kw1))
ax3.scatter(c1,0,color='k',s=160)
ax3.scatter([c1+delta*i/n for i in range(1,n)],[0 for i in range(1,n)],color=[['red','blue'][np.random.randint(2)] for i in range(1,n)],s=80)
ax3.add_patch(patches.FancyArrowPatch((c1-delta/2,1.3),(c1+delta/2,1.3),**kw1))
'''
for i in range(n-1):
    ax3.add_patch(patches.FancyArrowPatch((c1+delta*i/n,0),(c1+delta*(i+1)/n,0),connectionstyle="arc3,rad=1.2",**kw3))
'''
ax3.annotate(r'Discovery in a single direction',(c1,1.7),ha='center',fontsize=13,color='black')

ax3.add_patch(patches.FancyArrowPatch((c2-delta,0),(c2+delta,0),**kw1))
ax3.scatter(c2,0,color='k',s=160)
k = np.random.randint(2,n+1)
ax3.scatter([c2+delta*i/n for i in range(1,n)],[0 for i in range(1,n)],color=[['red','blue'][np.random.randint(2)] for i in range(1,n)],s=80)
ax3.scatter([c2-delta*i/n for i in range(1,k)],[0 for i in range(1,k)],color=[['red','blue'][np.random.randint(2)] for i in range(1,k)],s=80)
ax3.annotate(r'Discoveries possible in both directions',(c2,1.7),ha='center',fontsize=13,color='black')
ax3.add_patch(patches.FancyArrowPatch((c2-delta/2,1.3),(c2+delta/2,1.3),**kw1_double))
'''
ax3.add_patch(patches.FancyArrowPatch((c2,0),(c2-delta/n,0),connectionstyle="arc3,rad=1.2",**kw3))
ax3.add_patch(patches.FancyArrowPatch((c2-delta/n,0),(c2+delta/n,0),connectionstyle="arc3,rad=0.8",**kw3))
for i in range(2):
    ax3.add_patch(patches.FancyArrowPatch((c2+delta*(i+1)/n,0),(c2+delta*(i+2)/n,0),connectionstyle="arc3,rad=1.2",**kw3))
ax3.add_patch(patches.FancyArrowPatch((c2+3*delta/n,0),(c2-2*delta/n,0),connectionstyle="arc3,rad=0.4",**kw3))
ax3.add_patch(patches.FancyArrowPatch((c2-2*delta/n,0),(c2-3*delta/n,0),connectionstyle="arc3,rad=1.2",**kw3))
ax3.add_patch(patches.FancyArrowPatch((c2-3*delta/n,0),(c2+4*delta/n,0),connectionstyle="arc3,rad=0.4",**kw3))
for i in range(3):
    ax3.add_patch(patches.FancyArrowPatch((c2+(4+i)*delta/n,0),(c2+(5+i)*delta/n,0),connectionstyle="arc3,rad=1.2",**kw3))
'''
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.tick_params(axis='y', which='both', left=False, labelleft=False)
ax3.xaxis.set_ticks([])

ax3.set_title(r'\textbf{Deterministic versus random discovery order}',fontsize=15,y=1.25)

ytop = (ax3.title.get_window_extent().y0 + ax3.title.get_window_extent().y1)/10000
fig.text(0.08, ytop, r'\textbf{(a)}', fontsize=15,va='center')
ymiddle = (ax.title.get_window_extent().y0 + ax.title.get_window_extent().y1)/10000
fig.text(0.08, ymiddle, r'\textbf{(b)}', fontsize=15,va='center')
fig.text(0.52, ymiddle, r'\textbf{(c)}', fontsize=15,va='center')

plt.show()