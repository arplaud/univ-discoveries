import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec

mean_corr = np.zeros((1999,1999))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations brownien 1D') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations brownien 1D/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr = mean_corr - 1/4

list_frac = [[] for i in range(100)]
n = np.linspace(1,10**6,100)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided/' + file, allow_pickle = True)
        for i in range(100):
            list_frac[i] = np.concatenate((list_frac[i],f[i]))


list_var= [np.var(list_frac[i]) for i in range(100)]


n_max = 1999

list_c = [0.24949591554898765,0.22323266612181838,0.2053444010534825,0.19071778410088835,0.17816883934274022,0.16713023543809805,0.15726409512266276,0.14834549993346058,0.14021678341881189,0.13276635355697114,0.12591327886092715,0.11958958273677868,0.11372587847554667,0.10825562416957227,0.1031369463194081,0.09835659141373217,0.09389339163172745,0.08969620129697507,0.08573040198446266,
          0.08200554903098005,0.07850707954785675,0.07518434997690025,0.07204016402525047,0.06908514631320727,0.06626883325557253,0.06359471644255149,0.06107671244407861,0.05866333539112298,0.05637844280496961,0.05421207856147244,0.05213026490149073,0.05017048237968601,0.04828313207445729,0.04649377741523136,0.04478558164481572,0.04314636311684319,0.04159543980374236,0.040093862254878605,
          0.03867965468215029,0.037304270525705036,0.036010546469293796,0.034749516055505414,0.0335636659058571,0.032405507186891704,0.03131704000198018,0.030251583694580315,0.029250824894799667,0.028269976923983548,0.027347078797691184,0.02644521766384138,0.025589888239115753,0.02476314026143153,0.02396609053900448,0.02320952395624454,0.022465962415231237,0.021769737735990297,
          0.02108181199238687,0.020431556703496674,0.019803330911122016,0.019189707145872162,0.0186148921182458,0.01804356867739056,0.017503698840734723,0.016984422349284288,0.016471120234317262,0.015992918376454432,0.015520628287091385,0.015062870291489566,0.014634619156938415,0.014206054121940148,0.013797320580776404,0.013411579326333587,0.013023698614321424,0.01265675621804741,
          0.012308975409451813,0.0119582287197287,0.011626558228314562,0.01131332792864366,0.010996497463861893,0.010694411866547269,0.01041225042493552,0.010127215916129946,0.00985011634560641,0.009594455423376867,0.009340053854919424,0.009085563724895362,0.008850394067859268,0.008624580070392112,0.008393795212996788,0.008173638188527751,0.007970531196121577,0.007766391379831686,
          0.00756097021598553,0.007371159297275744,0.007191852706795542,0.007007689413409522,0.006826471486213428,0.0066610461413784475,0.0065016033744769355,0.006336642981748245]

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1,2, wspace=0.2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

ax1.set_xlim(10,1e6)
ax1.set_ylim(0.,0.08)
ax1.scatter(n,list_var,s=0.3,color='blue',label=r'$\mathrm{Var}[X_n]$')
ax1.hlines(0.03433,3e5,1e6,lw=0.5,color='red',label=r'theoretical value for $\mathrm{Var}[X_\infty]$')
ax1.legend(loc='upper right',markerscale=3,framealpha=1,facecolor='white', fontsize=8)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

axins = inset_axes(ax1,width="36%",height="36%",loc='lower center')
axins.set_xscale('log')
axins.set_yscale('log')

axins.scatter(n,[v-0.03433 for v in list_var],color='blue',s=0.1)
axins.plot(n,500/n,color='green',lw=0.4)
axins.set_xlim(1e5,1e6)
axins.set_ylim(3e-4,8e-2)

axins.set_xticks([])
axins.set_yticks([])

from matplotlib.ticker import NullFormatter

axins.xaxis.set_major_formatter(NullFormatter())
axins.xaxis.set_minor_formatter(NullFormatter())

axins.yaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_minor_formatter(NullFormatter())

ax2.scatter([j/(i+1) for j in range(20,n_max//2,5) for i in range(100,n_max//2,20)],[real_corr[i,i+j] for j in range(20,n_max//2,5) for i in range(100,n_max//2,20)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax2.plot(np.linspace(0,1,100),np.array(list_c),c='red',lw=0.5,label=r'theoretical expression for $c(r)$')
ax2.set_xlim(0,1)
ax2.set_ylim(0,3*10**-1)
ax2.legend(loc='upper right',markerscale= 3,framealpha=1, facecolor='white', fontsize=8)
##plt.yscale('log')
ax2.set_xlabel(r'$l/k$',fontsize=11)

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

list_frac = []

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Brownien 1D/' + file, allow_pickle = True)
        list_frac = np.concatenate((list_frac,f))

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1,2,wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

x = np.linspace(0,1,1000)

ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.hist(list_frac,density=True,cumulative=True,bins=1000,color='blue',histtype='step',label=r'CDF of $X_\infty$',lw=0.2)
ax1.set_xlabel(r'$x$',fontsize=11)
ax1.plot(x,beta.cdf(x,3.14,3.14),color='red',lw=0.2,label=r'Beta prediction')
ax1.legend(loc='upper left',markerscale=3,framealpha=1,facecolor='white',fontsize=8)

axins = inset_axes(ax1,width="36%",height="36%",loc='lower right')
axins.set_xscale('log')
axins.set_yscale('log')

axins.hist(list_frac,density=True,cumulative=True,bins=10000,color='blue',histtype='step',lw=0.5)

axins.plot(x,beta.cdf(x,3.14,3.14),color='red',lw=0.8)

# Zone zoomée très proche de x = 0
axins.set_xlim(10**(-2),0.1)
axins.set_ylim(2e-5,0.005)

# Relie l'inset à la zone correspondante
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

list_frac = [[] for i in range(100)]
list_frac_ref = [[] for i in range(100)]
n = np.linspace(1,10**6,100)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D two-sided/' + file, allow_pickle = True)
        for i in range(100):
            list_frac[i] = np.concatenate((list_frac[i],f[i]))


for file in os.listdir('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D one-sided'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/autocovariance fraction 1D one-sided/' + file, allow_pickle = True)
        for i in range(100):
            list_frac_ref[i] = np.concatenate((list_frac_ref[i],f[i]))



list_beta_test= [np.mean((list_frac[i]-1/2)**4)/np.mean((list_frac[i]-1/2)**2)**2 - 3/(1+8*np.mean((list_frac[i]-1/2)**2)) for i in range(100)]
list_beta_test_ref= [np.mean((list_frac_ref[i]-1/2)**4)/np.mean((list_frac_ref[i]-1/2)**2)**2 - 3/(1+8*np.mean((list_frac_ref[i]-1/2)**2)) for i in range(100)]


ax2.set_xlim(2e5,1e6)
ax2.set_ylim(-0.015,0.1)
ax2.set_xlabel(r'$n$',fontsize=11)
ax2.scatter(n,list_beta_test,s=0.3,color='blue',label=r'$\frac{\hat{m}_4}{\hat{m}_2^2} - \frac{3}{1+8 \hat{m}_2}$')
ax2.scatter(n,list_beta_test_ref,s=0.3,color='green',label=r'$\frac{\hat{m}_4}{\hat{m}_2^2} - \frac{3}{1+8 \hat{m}_2}$ (one-sided)')
'''
ax2.set_yscale('log')
ax2.set_xscale('log')
'''
ax2.hlines(0,5e5,1e6,color='red',lw=0.5,label=r'symmetric Beta value')
ax2.legend(loc='upper left',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

# Optionnel : retirer les ticks
axins.set_xticks([])
axins.set_yticks([])

from matplotlib.ticker import NullFormatter

axins.xaxis.set_major_formatter(NullFormatter())
axins.xaxis.set_minor_formatter(NullFormatter())

axins.yaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_minor_formatter(NullFormatter())

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec


mean_corr = np.zeros((9282,9282))
compteur = 0


for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 1.5') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha = 1.5/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_15 = mean_corr - 1/4


mean_corr = np.zeros((1999,1999))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.25 new') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.25 new/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_125 = mean_corr - 1/4
    
fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1,2,wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_max=9282
ax1.scatter([j/(i+1) for j in range(10,n_max//2,3) for i in range(100,n_max//2,30)],[real_corr_15[i,i+j] for j in range(10,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax1.set_xlabel(r'$l/k$',fontsize=11)
ax1.set_xlim(0,1)
ax1.set_ylim(0,0.25)
ax1.set_title(r'$\alpha=3/2$',y=0.7,fontsize=12)
ax1.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

n_max=1999
ax2.scatter([j/(i+1) for j in range(4,n_max//2,3) for i in range(40,n_max//2,10)],[real_corr_125[i,i+j] for j in range(4,n_max//2,3) for i in range(40,n_max//2,10)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax2.set_xlabel(r'$l/k$',fontsize=11)
ax2.set_xlim(0,1)
ax2.set_ylim(0,0.2)
ax2.set_title(r'$\alpha=5/4$',y=0.7,fontsize=12)
ax2.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')
    
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

list_frac_2DBM = [[] for i in range(50)]
n1 = np.logspace(1,6,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Brownien 2D N=10 to 10^6'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Brownien 2D N=10 to 10^6/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_2DBM[i] = np.concatenate((list_frac_2DBM[i],f[i]))


list_var_2DBM= [np.var(list_frac_2DBM[i]) for i in range(50)]

def f_fit_2DBM(x,a,b):
    return(0.33/np.log(x)**2 - a/np.log(x)**3 + b/np.log(x)**4)

a2DBM, b2DBM = curve_fit(f_fit_2DBM,n1[30:],list_var_2DBM[30:])[0]

list_frac_cauchy = [[] for i in range(50)]
n2 = np.logspace(1,6,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 1 N=10 to 10^6'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction levy alpha = 1 N=10 to 10^6/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_cauchy[i] = np.concatenate((list_frac_cauchy[i],f[i]))


list_var_cauchy= [np.var(list_frac_cauchy[i]) for i in range(50)]

def f_fit_Cauchy(x,a,b):
    return(0.3465/np.log(x)**2 - a/np.log(x)**3 + b/np.log(x)**4)

aCauchy, bCauchy = curve_fit(f_fit_Cauchy,n2[30:],list_var_cauchy[30:])[0]

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1,2,wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


ax1.scatter(n1,list_var_2DBM,s=0.3,color='blue',label = r'$\mathrm{Var}[X_n]$')
ax1.plot(n1[20:],f_fit_2DBM(n1[20:],a2DBM,b2DBM),lw=0.3,color='green',label='theoretical prediction with corrections')
ax1.plot(n1[20:],0.33/np.log(n1[20:])**2,lw=0.3,color='red',label=r'theoretical prediction at leading order')
ax1.set_xlabel(r'$n$',fontsize=11)
ax1.set_title(r'$2$D Brownian motion',fontsize=12,y=0.65)
ax1.set_ylim(0.,0.02)
ax1.set_xlim(10,1.2e6)
ax1.set_xscale('log')
ax1.legend(loc='upper right',framealpha=1,facecolor='white',markerscale=3,fontsize=6.6)


ax2.scatter(n2,list_var_cauchy,s=0.4,color='blue',label = r'$\mathrm{Var}[X_n]$')
ax2.plot(n2[20:],f_fit_Cauchy(n2[20:],aCauchy,bCauchy),lw=0.5,color='green',label=r'theoretical prediction with corrections')
ax2.plot(n2[20:],0.3465/np.log(n2[20:])**2,lw=0.3,color='red',label='theoretical prediction at leading order')
ax2.set_xlabel(r'$n$',fontsize=11)
ax2.set_title(r'$1$D Cauchy walk',fontsize=12,y=0.65)
ax2.set_xscale('log')
ax2.set_ylim(0,0.02)
ax2.set_xlim(10,1.2e6)
ax2.legend(loc='upper right',framealpha=1,facecolor='white',markerscale=3,fontsize=6.6)

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
    
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

list_frac = [[] for i in range(50)]
n = np.logspace(0,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.75'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction Levy 1D alpha = 0.75/' + file, allow_pickle = True)
        for i in range(50):
            list_frac[i] = np.concatenate((list_frac[i],f[i]))


list_var= [np.var(list_frac[i]) for i in range(50)]

def f_fit(x,a):
    return(0.1266/x**(2/3) - a/x)

a = curve_fit(f_fit,n[30:],list_var[30:])[0]

mean_corr = np.zeros((199,199))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha=0.75') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha=0.75/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr = mean_corr - 1/4

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1,2,wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

ax1.scatter(n,list_var,s=0.3,color='blue',label=r'$\mathrm{Var}[X_n]$')
ax1.plot(n[20:],0.1266/n[20:]**(2/3),lw=0.3,color='red',label=r'theoretical decay')
##ax1.plot(n[20:],f_fit(n[20:],a),color='green',lw=0.3,label=r'theoretical prediction with correction')
ax1.set_xlabel(r'$n$',fontsize=11)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(10,1.2e7)
ax1.legend(loc='upper right',framealpha=1,facecolor='white',markerscale=3,fontsize=8)

list_c = [0.047963, 0.0436509, 0.0406999, 0.038403, 0.0365043, 0.0348783, \
0.0334528, 0.0321821, 0.0310352, 0.02999, 0.0290298, 0.0281422, \
0.0273173, 0.026547, 0.0258251, 0.025146, 0.0245054, 0.0238995, \
0.023325, 0.0227792, 0.0222595, 0.0217639, 0.0212905, 0.0208377, \
0.0204039, 0.0199878, 0.0195882, 0.0192041, 0.0188345, 0.0184784, \
0.0181352, 0.017804, 0.0174842, 0.0171751, 0.0168762, 0.016587, \
0.0163069, 0.0160355, 0.0157724, 0.0155171, 0.0152694, 0.0150288, \
0.0147951, 0.014568, 0.0143471, 0.0141322, 0.0139231, 0.0137194, \
0.0135211, 0.0133279, 0.0131395, 0.0129559, 0.0127767, 0.012602, \
0.0124314, 0.0122648, 0.0121022, 0.0119433, 0.0117881, 0.0116364, \
0.011488, 0.011343, 0.0112011, 0.0110622, 0.0109264, 0.0107934, \
0.0106632, 0.0105357, 0.0104108, 0.0102885, 0.0101686, 0.0100511, \
0.00993598, 0.00982306, 0.00971234, 0.00960374, 0.00949721, \
0.00939268, 0.00929011, 0.00918944, 0.00909062, 0.0089936, \
0.00889833, 0.00880477, 0.00871287, 0.00862259, 0.00853389, \
0.00844673, 0.00836106, 0.00827686, 0.00819409, 0.0081127, \
0.00803267, 0.00795397, 0.00787657, 0.00780042, 0.00772551, \
0.0076518, 0.00757927, 0.00750789, 0.00743763, 0.00736848, \
0.00730039, 0.00723336, 0.00716735, 0.00710234, 0.00703832, \
0.00697526, 0.00691314, 0.00685194, 0.00679164, 0.00673223, \
0.00667367, 0.00661597, 0.00655909, 0.00650302, 0.00644775, \
0.00639326, 0.00633953, 0.00628655, 0.0062343, 0.00618277, \
0.00613194, 0.00608181, 0.00603235, 0.00598356, 0.00593542, \
0.00588793, 0.00584105, 0.0057948, 0.00574915, 0.00570409, \
0.00565962, 0.00561571, 0.00557237, 0.00552958, 0.00548733, \
0.00544562, 0.00540442, 0.00536374, 0.00532356, 0.00528388, \
0.00524468, 0.00520596, 0.00516772, 0.00512993, 0.0050926, \
0.00505571, 0.00501926, 0.00498325, 0.00494765, 0.00491248, \
0.00487771, 0.00484335, 0.00480939, 0.00477582, 0.00474263, \
0.00470982, 0.00467738, 0.0046453, 0.00461359, 0.00458223, \
0.00455122, 0.00452056, 0.00449023, 0.00446023, 0.00443057, \
0.00440122, 0.0043722, 0.00434348, 0.00431508, 0.00428698, \
0.00425917, 0.00423167, 0.00420445, 0.00417752, 0.00415087, \
0.0041245, 0.0040984, 0.00407257, 0.00404701, 0.0040217, 0.00399666, \
0.00397187, 0.00394733, 0.00392304, 0.003899, 0.00387519, 0.00385162, \
0.00382828, 0.00380518, 0.0037823, 0.00375964, 0.00373721, 0.003715, \
0.003693, 0.00367121, 0.00364963, 0.00362826, 0.00360709]

n_max=199
ax2.scatter([j/(i+1) for j in range(3,n_max//2) for i in range(10,n_max//2,3)],[j**(2/3)*real_corr[i,i+j] for j in range(3,n_max//2) for i in range(10,n_max//2,3)],s=0.3,c='blue',label = r'$l^{2/3} Cov(I_k,I_{k+l})$')
ax2.set_xlim(0,2)
ax2.set_ylim(0,0.04)
ax2.plot(np.linspace(0,2,200),list_c,lw=0.3,color='r',label=r'theoretical expression for $c(r)$')
ax2.set_xlabel('$l/k$')
ax2.legend(framealpha=1,facecolor='white',loc='upper right',markerscale=3,fontsize=8)

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

list_frac = [[] for i in range(50)]
n = np.logspace(1,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM N=10 to 10^7'):
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM N=10 to 10^7/' + file, allow_pickle = True)
        for i in range(50):
            list_frac[i] = np.concatenate((list_frac[i],f[i]))


list_var= [np.var(list_frac[i]) for i in range(50)]

def f_fit(x,a):
    return((0.098*np.log(x)+a)/x)

a = curve_fit(f_fit,n[40:],list_var[40:])[0]

fig = plt.figure(figsize=(5,3))
gs = gridspec.GridSpec(1,1,wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
#ax2 = fig.add_subplot(gs[0,1])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

ax1.scatter(n,list_var,s=0.3,color='blue',label=r'$\mathrm{Var}[X_n]$')
ax1.plot(n[20:],0.098*np.log(n[20:])/n[20:],lw=0.3,color='red',label=r'theoretical leading order')
ax1.plot(n[20:],f_fit(n[20:],a),lw=0.3,color='green',label=r'theoretical prediction with correction')
ax1.set_xlabel(r'$n$',fontsize=11)
ax1.set_xlim(10,1.2e7)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(framealpha=1,facecolor='white',loc='upper right',fontsize=8,markerscale=3)

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec


mean_corr = np.zeros((1998,1998))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.5 hetero') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.5 hetero/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_15 = mean_corr -0.1897
    
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_max=1998
ax1.scatter([j/(i+1) for j in range(1,n_max//2,3) for i in range(100,n_max//2,30)],[real_corr_15[i,i+j] for j in range(1,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax1.set_xlabel(r'$l/k$',fontsize=11)
ax1.set_xlim(0,1)
ax1.set_ylim(0,0.25)
ax1.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

    
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec


mean_corr = np.zeros((1998,1998))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.5 n=7') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations levy alpha=1.5 n=7/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_15 = mean_corr - 1/49
    
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_max=1998
ax1.scatter([j/(i+1) for j in range(1,n_max//2,3) for i in range(100,n_max//2,30)],[real_corr_15[i,i+j] for j in range(1,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax1.set_xlabel(r'$l/k$',fontsize=11)
ax1.set_xlim(0,1)
ax1.set_ylim(0,0.25)
ax1.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

    
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec


mean_corr = np.zeros((1998,1998))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations percolation') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations percolation/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_15 = mean_corr - 1/4
    
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_max=1998
ax1.scatter([j/(i+1) for j in range(3,n_max//2,3) for i in range(100,n_max//2,30)],[real_corr_15[i,i+j] for j in range(3,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='blue',label = r'$Cov(I_k,I_{k+l})$')
ax1.set_xlabel(r'$l/k$',fontsize=11)
ax1.set_xlim(0,1)
ax1.set_ylim(0,0.25)
ax1.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

    
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec


mean_corr = np.zeros((1998,1998))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations fBm 2D H=0.33') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations fBm 2D H=0.33/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_15 = mean_corr - 1/4

mean_corr = np.zeros((1998,1998))
compteur = 0

for file in os.listdir('/users/invites/plaud/Documents/range_competition/correlations fBm 2D H=0.66') :
    if os.path.splitext(file)[-1] == '.npy':
        corr= np.load('/users/invites/plaud/Documents/range_competition/correlations fBm 2D H=0.66/' + file, allow_pickle = True)
        compteur += 1
        mean_corr+= corr
    
mean_corr = mean_corr/compteur

real_corr_05 = mean_corr - 1/4
    
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

n_max=1998
ax1.scatter([j/(i+1) for j in range(4,n_max//2,3) for i in range(100,n_max//2,30)],[real_corr_15[i,i+j] for j in range(4,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='blue',label = r'$\mathrm{Cov}(I_k,I_{k+l})$')
ax1.set_xlabel(r'$l/k$',fontsize=11)
'''
ax1.set_ylabel(r'$L(l)^{-1} \mathrm{Cov}(I_k,I_{k+l})$')
n_max=1998

ax1.scatter([j/(i+1) for j in range(30,n_max//2,3) for i in range(100,n_max//2,30)],[j**(2/3)*real_corr_05[i,i+j] for j in range(30,n_max//2,3) for i in range(100,n_max//2,30)],s=0.3,c='red',label = r'$H=2/3$')
'''
ax1.set_xlim(0,0.5)
ax1.set_ylim(0,0.25)
ax1.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,facecolor='white')

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])
plt.rcParams["figure.dpi"] = 1000
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

list_frac_1 = [[] for i in range(50)]
n = np.logspace(1,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur6 vanishing') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur6 vanishing/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_1[i] = np.concatenate((list_frac_1[i],f[i]))


list_var= [np.var(list_frac_1[i]) for i in range(50)]


ax1.scatter(n,list_var,s=0.3,c='blue',label=r'asymptotically vanishing cone $\theta=\pi/6$')

list_frac_2 = [[] for i in range(50)]
n = np.logspace(1,6,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur8') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur8/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_2[i] = np.concatenate((list_frac_2[i],f[i]))


list_var= [np.var(list_frac_2[i]) for i in range(50)]
ax1.scatter(n,list_var,s=0.3,c='green',label=r'infinite cone $\theta = \pi/8$')

list_frac_3 = [[] for i in range(50)]
n = np.logspace(1,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM N=10 to 10^7') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM N=10 to 10^7/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_3[i] = np.concatenate((list_frac_3[i],f[i]))


list_var= [np.var(list_frac_3[i]) for i in range(50)]

ax1.scatter(n,list_var,s=0.3,c='red',label=r'normal 3D simple random walk')

list_frac_4 = [[] for i in range(50)]
n = np.logspace(1,6,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur12') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur12/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_4[i] = np.concatenate((list_frac_4[i],f[i]))


list_var= [np.var(list_frac_4[i]) for i in range(50)]

ax1.scatter(n,list_var,s=0.3,c='purple',label=r'infinite cone $\theta=\pi/12$')

list_frac_5 = [[] for i in range(50)]
n = np.logspace(1,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM odd planes') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM odd planes/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_5[i] = np.concatenate((list_frac_5[i],f[i]))


list_var= [np.var(list_frac_5[i]) for i in range(50)]

ax1.scatter(n,list_var,s=0.3,c='orange',label=r'odd planes')

list_frac_6 = [[] for i in range(50)]
n = np.logspace(1,7,50)

for file in os.listdir('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur3 fini') :
    if os.path.splitext(file)[-1] == '.npy':
        f = np.load('/users/invites/plaud/Documents/range_competition/fraction 3D BM theta = pisur3 fini/' + file, allow_pickle = True)
        for i in range(50):
            list_frac_6[i] = np.concatenate((list_frac_6[i],f[i]))


list_var= [np.var(list_frac_6[i]) for i in range(50)]

ax1.scatter(n,list_var,s=0.3,c='pink',label=r'finite cone $\theta = \pi/3$')

ax1.set_ylim(5*10**-8,0.1)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\mathrm{Var}[X_n]$')
ax1.set_xlabel(r'$n$')
ax1.legend(loc='lower left',markerscale=3,fontsize=6,framealpha=1,facecolor='white')