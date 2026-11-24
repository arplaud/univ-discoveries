import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import kurtosis


list_frac_inter = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.75') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.75/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_inter[i] = np.concatenate((list_frac_inter[i],f[i]))

list_var_inter= [np.var(list_frac_inter[i]) for i in range(50)]

list_frac_recu = [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_recu[i] = np.concatenate((list_frac_recu[i],f[i]))

list_var_recu= [np.var(list_frac_recu[i]) for i in range(50)]

list_frac_trans= [[] for i in range(50)]

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.5') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.5/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_trans[i] = np.concatenate((list_frac_trans[i],f[i]))    
          
list_var_trans = [np.var(list_frac_trans[i]) for i in range(50)]



def f(x,a,b):
    return(a*x**(-2/3) - b*x**(-1))

n1 = np.logspace(0,6,50)
n2 = np.logspace(0,7,50)
n3 = np.logspace(0,6,50)

a,b = curve_fit(f,n2[30:40],list_var_inter[30:40])[0]

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[2,2,2], hspace=0.6, wspace=0.3)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['figure.dpi']=1000

ax = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax2prime = fig.add_subplot(gs[1,2])
ax3 = fig.add_subplot(gs[0, :])

# -----------------------------
# Panel (b) : Variance decay
# -----------------------------
ax.scatter(n3[10:], list_var_recu[10:], color='b', s=3, label=r'$d_s=1$')
ax.plot(n2[22:47], 0.03433/n2[22:47]**(0), color='b', lw=1)
ax.annotate(r'$\sim \mathrm{Var}(X_{\infty})$', (3*10**4, 0.006), color='b', fontsize=15)

ax.scatter(n2[9:], list_var_inter[9:], color='r', s=3, label=r'$d_s=8/3$')
ax.plot(n2[22:], a/n2[22:]**(2/3), color='r', lw=1)
ax.annotate(r'$\sim n^{-(d_s-2)}$', (1.3*10**5, 8e-5), color='r', fontsize=15)

ax.scatter(n1[12:], list_var_trans[12:], color='g', s=3, label=r'$d_s=4$')
ax.plot(n1[22:], 0.12/n1[22:]**(1), color='g', lw=1)
ax.annotate(r'$\sim n^{-1}$', (7*10**5, 3e-7), color='g', fontsize=15)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'number of discovered sites $n$', fontsize=13)
ax.set_ylabel(r'$\mathrm{Var}(X_n)$', fontsize=13)
ax.yaxis.set_label_coords(-0.16, 0.5)
ax.set_xlim(10, 2*10**7)
ax.set_ylim(6e-8, 5e-2)

ax.legend(framealpha=1, facecolor='white', markerscale=2, fontsize=10.8)
ax.set_title(r'\textbf{Discovery-share \\ fluctuation regimes}', fontsize=15,y=1.08,va='center')

ax.grid(False)
# -----------------------------
# Panel (c) : Structure of correlations
# -----------------------------

n=1999

mean_corr_recu = np.zeros((n,n))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations brownien 1D') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations brownien 1D/' + file, allow_pickle = True)
        compteur += 1
        mean_corr_recu+= corr
    
mean_corr_recu = mean_corr_recu/compteur

real_corr_recu = mean_corr_recu - 1/4


list_i = np.floor(np.logspace(0,np.log10(n//1.4),30)).astype('int')
ax2.scatter(list_i,[real_corr_recu[i,int(1.3*i)] for i in list_i],color='b',s=3, label = r'$d_s = 1$')
ax2.plot(list_i[15:],0.05/list_i[15:]**0,color='b',lw=1)
ax2.annotate(r'$\sim c(0.3)$',(3e2,2.3e-2),color='b',fontsize=15)

n=1999
mean_corr_inter = np.zeros((n,n))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.75 new') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.75 new/' + file, allow_pickle = True)
        compteur += 1
        mean_corr_inter+= corr
    
mean_corr_inter = mean_corr_inter/compteur

real_corr_inter = mean_corr_inter - 1/4

list_i = np.floor(np.logspace(0,np.log10(n//1.4),30)).astype('int')
ax2.scatter(list_i,[real_corr_inter[i,int(1.3*i)] for i in list_i],color='r',s=3, label = r'$d_s = 8/3$')
ax2.plot(list_i[15:],0.035/list_i[15:]**(2/3),color='r',lw=1)
ax2.annotate(r'$\sim c(0.3) \ell^{-(d_s-2)}$',(0.8e2,3e-3),color='r',fontsize=15)

n=1999
mean_corr_trans = np.zeros((n,n))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.66') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.66/' + file, allow_pickle = True)
        compteur += 1
        mean_corr_trans+= corr
    
mean_corr_trans = mean_corr_trans/compteur

real_corr_trans = mean_corr_trans - 1/4

list_i = np.floor(np.logspace(0,np.log10(n//1.4),30)).astype('int')
ax2.scatter(list_i,[real_corr_trans[i,int(1.3*i)] for i in list_i],color='g',s=3, label = r'$d_s = 3$')
ax2.plot(list_i[15:],0.08/list_i[15:]**(1),color='g',lw=1)
ax2.annotate(r'$\sim c(0.3) \ell^{-(d_s-2)}$',(0.9e2,1.6e-5),color='g',fontsize=15)

n=199
mean_corr_trans_high = np.zeros((n,n))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.66 high stat') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 0.66 high stat/' + file, allow_pickle = True)
        compteur += 1
        mean_corr_trans_high+= corr
    
mean_corr_trans_high = mean_corr_trans_high/compteur

real_corr_trans_high = mean_corr_trans_high - 1/4

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'discovery label $\ell$', fontsize=13)
ax2.set_ylabel(r'$\mathrm{Cov}(I_\ell,I_{1.3\ell})$', fontsize=13)
ax2.yaxis.set_label_coords(-0.16, 0.5)
ax2.set_xlim(6, 1.9*10**3)
ax2.set_ylim(8e-6,0.9e-1)

ax2.legend(framealpha=1, facecolor='white', fontsize=10.8,markerscale=2)
ax2.set_title(r'\textbf{Discovery-label \\ correlation decay}', fontsize=15,y=1.08, va='center')

ax2.grid(False)

n=1999
ax2prime.scatter([j/(i+1) for j in range(20,n//2,10) for i in range(20,n//2,100)],[real_corr_recu[i,i+j] for j in range(20,n//2,10) for i in range(20,n//2,100)],color='b',s=0.1,label=r'$d_s = 1$')

n=200
ax2prime.scatter([j/(i+1) for j in range(3,n//2,2) for i in range(10,n//2,10)],[j**(2/3)*real_corr_inter[i,i+j] for j in range(3,n//2,2) for i in range(10,n//2,10)],color='r',s=0.1,label=r'$d_s = 8/3$')

'''
n=199
ax2prime.scatter([j/(i+1) for j in range(20,n//2,5) for i in range(20,n//2,5)],[j**(1)*real_corr_trans_high[i,i+j] for j in range(20,n//2,5) for i in range(20,n//2,5)],color='g',s=0.1,label=r'$d_s = 3$')
'''

ax2prime.set_yscale('log')
ax2prime.set_xlabel(r'aging ratio $\ell/k$', fontsize=13)
ax2prime.set_ylabel(r'$L(\ell)^{-1} \mathrm{Cov}(I_k,I_{k+\ell})$', fontsize=13)
ax2prime.yaxis.set_label_coords(-0.16, 0.5)
ax2prime.set_xlim(0, 1)
ax2prime.set_ylim(1e-3,0.3)

ax2prime.legend(framealpha=1, facecolor='white', fontsize=10.8,markerscale=np.sqrt(120),loc='lower left')
ax2prime.set_title(r'\textbf{Aging correlation collapse}', fontsize=15,y=1.08)

ax2prime.grid(False)

# Panel (a) : Phase Diagram

ax3.set_xlim(1,4)
ax3.set_ylim(0,1.2)
ax3.vlines((2,3),0,1,color='black',lw=2.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.tick_params(axis='y', which='both', left=False, labelleft=False)
ax3.xaxis.set_ticks([])
ax3.set_xlabel(r'spectral dimension $d_s$',fontsize=13,labelpad=5)
ax3.fill_between([1,2],0,0.9,color='b',alpha=0.2)
ax3.fill_between([2,3],0,0.9,color='r',alpha=0.2)
ax3.fill_between([3,4],0,0.9,color='g',alpha=0.2)
ax3.annotate(r'\textbf{Persistent randomness}',(1.5,0.4),fontsize=15,ha='center')
ax3.annotate(r'\begin{center}\textbf{Anomalous non-Gaussian \\ concentration} \end{center}',(2.5,0.4),fontsize=15,ha='center',va='center')
ax3.annotate(r'\textbf{Gaussian concentration}',(3.5,0.4),fontsize=15,ha='center')
ax3.annotate(r'{\boldmath $d_s=2$}',(2,1.1),fontsize=14,ha='center')
ax3.annotate(r'{\boldmath $d_s=3$}',(3,1.1),fontsize=14,ha='center')
ax3.set_title(r'\textbf{Universal phase diagram}',fontsize=15,y=1.15)

ytop = (ax3.title.get_window_extent().y0 + ax3.title.get_window_extent().y1)/10000
ymiddle = (ax2prime.title.get_window_extent().y0 + ax2prime.title.get_window_extent().y1)/10000

fig.text(0.09, ytop, r'\textbf{(a)}', fontsize=15,va='center')
fig.text(0.09, ymiddle, r'\textbf{(b)}', fontsize=15,va='center')
fig.text(0.37, ymiddle, r'\textbf{(c)}', fontsize=15,va='center')
fig.text(0.66, ymiddle, r'\textbf{(d)}', fontsize=15, va='center')

plt.show()