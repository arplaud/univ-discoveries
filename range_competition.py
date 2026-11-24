import numpy as np
##from numba import jit
from time import time
import matplotlib.pyplot as plt
from scipy.stats import levy
from scipy.stats import cauchy
from scipy.stats import levy_stable
from scipy.stats import norm
import sys


# index=int(sys.argv[1])

D = 0.5
dt = 1
dx = 0.01
diffusion_ratio = 1
drift= 0.
discretisation_record = 10**5


##@jit(nopython = True)
def range_values(t_max,n = 2,proba_negative = 0,alpha = 2):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,n))
    current_max = 0
    current_min = 0
    ranges = np.zeros(n)
    range_tot = 0
    steps = 0
    recordman = 0
    n_max = 0.05*t_max**(1/alpha)
    while steps < N and range_tot < n_max*discretisation_record:
        if alpha == 2:
            steps_x = np.random.normal(0.,np.sqrt(2*D*dt),n)
        else:
            steps_x = levy_stable.rvs(alpha,0,scale = (2*D*dt)**(1/alpha),size = n)
        for i in range(n):
            pos_x[steps+1,i] = pos_x[steps,i] + steps_x[i]
        max_pos, min_pos = 0,0
        iM, im = 0,0
        for i,pos in enumerate(pos_x[steps+1]):
            if pos > max_pos:
                iM, max_pos = i,pos
            if pos < min_pos:
                im, min_pos = i,pos
        if max_pos > current_max:
            beaten =  int((max_pos-current_max)*discretisation_record)
            ranges[iM] += beaten
            range_tot += beaten
            current_max = int(discretisation_record*max_pos)/discretisation_record
            recordman = iM
        if min_pos < current_min:
                beaten =  int(proba_negative*(current_min - min_pos)*discretisation_record)
                ranges[im] += beaten
                range_tot += beaten
                current_min = int(discretisation_record*min_pos)/discretisation_record
                recordman = im
        steps+=1
    final_pos = pos_x[steps,recordman]
    other_pos = np.delete(pos_x[steps],recordman)
    mean_gap = np.mean(pos_x) - final_pos
    gaps = np.array([abs(x - final_pos) for x in other_pos])
    if range_tot >0:
        return(ranges[0]/range_tot,range_tot/(t_max**(1/alpha)*discretisation_record))
    else:
        x = np.random.random()
        return(x,0)
    
def autocorrelation_fraction(t_max,n = 2,proba_negative = 0,alpha = 2):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,n))
    current_max = 0
    current_min = 0
    ranges = np.zeros(n)
    range_tot = 0
    steps = 0
    n_max = 0.05*t_max**(1/alpha)
    list_fractions = np.zeros(100)
    for j in range(100):
        while steps < (j+1)*N/100 and range_tot < (j+1)*n_max*discretisation_record/100:
            if alpha == 2:
                steps_x = np.random.normal(0.,np.sqrt(2*D*dt),n)
            else:
                steps_x = levy_stable.rvs(alpha,0,scale = (2*D*dt)**(1/alpha),size = n)
            for i in range(n):
                pos_x[steps+1,i] = pos_x[steps,i] + steps_x[i]
            max_pos, min_pos = 0,0
            iM, im = 0,0
            for i,pos in enumerate(pos_x[steps+1]):
                if pos > max_pos:
                    iM, max_pos = i,pos
                if pos < min_pos:
                    im, min_pos = i,pos
            if max_pos > current_max:
                beaten =  int((max_pos-current_max)*discretisation_record)
                ranges[iM] += beaten
                range_tot += beaten
                current_max = int(discretisation_record*max_pos)/discretisation_record
            if min_pos < current_min:
                beaten =  int(proba_negative*(current_min - min_pos)*discretisation_record)
                ranges[im] += beaten
                range_tot += beaten
                current_min = int(discretisation_record*min_pos)/discretisation_record
            steps+=1
        list_fractions[j] = ranges[0]/range_tot
    return(list_fractions)
    
def competition_levy(t_max,n_max,n=2,proba_negative=1,alpha=1):
    pos_x = np.zeros((t_max+1,n))
    visited_sites = {0}
    ranges = np.zeros(n)
    range_tot = 1
    steps = 0
    list_labels=[]
    while steps < t_max and range_tot < n_max:
        i = np.random.choice(range(n))
        d = (2*int(2*np.random.random()) - 1)*np.random.zipf(alpha+1)
        for j in range(n):
            pos_x[steps+1,j] = pos_x[steps,j]
        pos_x[steps+1,i] += d
        if pos_x[steps+1,i] not in visited_sites:
            ranges[i]+=1
            range_tot+=1
            visited_sites.add(pos_x[steps+1,i])
            list_labels.append(i)
        steps+=1
    if range_tot==n_max:
        return(True,ranges[0]/range_tot,list_labels)
    else:
        return(False,np.random.random(),[])
    

def competition_levy_fast(t_max, n_max, n=2, proba_negative=1, alpha=1,betas=None):
    
    if betas == None:
        betas = np.zeros(n)
    pos = np.zeros(n, dtype=np.int64)
    visited_sites = {0}
    ranges = np.zeros(n, dtype=np.int64)

    range_tot = 1
    list_labels = []

    # ===== PRE-GENERATION =====
    indices = np.random.randint(0, n, size=t_max)
    signs = np.random.randint(0, 2, size=t_max) * 2 - 1
    jumps = np.random.zipf(alpha + 1, size=t_max)

    for step in range(t_max):

        i = indices[step]
        pos[i] += int((1+betas[i]) * signs[step] * jumps[step])
        site = pos[i]

        if site not in visited_sites:
            visited_sites.add(site)
            ranges[i] += 1
            range_tot += 1
            list_labels.append(i)

            if range_tot == n_max:
                return (True, ranges[0] / range_tot, list_labels,abs(pos[1]-pos[0])/n_max)

    return (False, np.random.random(), [],0)
    

def range_competition_team(t_max,n=1,proba_negative = 0,alpha=2):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,2*n))
    current_max = 0
    current_min = 0
    ranges = [0,0]
    range_tot = 0
    steps = 0
    recordman = 0
    n_max = 0.05*t_max**(1/alpha)
    while steps < N and range_tot < n_max*discretisation_record:
        if alpha == 2:
            steps_x = np.random.normal(0.,np.sqrt(2*D*dt),2*n)
        else:
            steps_x = levy_stable.rvs(alpha,0,scale = (2*D*dt)**(1/alpha),size = 2*n)
        for i in range(2*n):
            pos_x[steps+1,i] = pos_x[steps,i] + steps_x[i]
        max_pos, min_pos = 0,0
        iM, im = 0,0
        for i,pos in enumerate(pos_x[steps+1]):
            if pos > max_pos:
                iM, max_pos = i,pos
            if pos < min_pos:
                im, min_pos = i,pos
        if max_pos > current_max:
            beaten =  int((max_pos-current_max)*discretisation_record)
            ranges[iM//n] += beaten
            range_tot += beaten
            current_max = int(discretisation_record*max_pos)/discretisation_record
            recordman = iM
        if min_pos < current_min:
                beaten =  int(proba_negative*(current_min - min_pos)*discretisation_record)
                ranges[im//n] += beaten
                range_tot += beaten
                current_min = int(discretisation_record*min_pos)/discretisation_record
                recordman = im
        steps+=1
    final_pos = pos_x[steps,recordman]
    other_pos = np.delete(pos_x[steps],recordman)
    mean_gap = np.mean(pos_x) - final_pos
    gaps = np.array([abs(x - final_pos) for x in other_pos])
    if range_tot >0:
        return(ranges[0]/range_tot,range_tot/(t_max**(1/alpha)*discretisation_record))
    else:
        x = np.random.random()
        return(x,0)
    
def autocorrelation_jump_processes(alpha=2,t_max=10**5):
    N = int(t_max/dt)
    steps = levy_stable.rvs(alpha = alpha,beta=0,scale = dt,size = N)
    list_positions = np.cumsum(steps)
    plus_time = np.zeros(N+1)
    for i,t in enumerate(list_positions):
        if t >0:
            plus_time[i+1] = plus_time[i]+1
        else:
            plus_time[i+1] = plus_time[i]
    i0 = N//10
    list_correlations = [plus_time[i0]*(plus_time[i0 + i*i0//10] - plus_time[i0])/(i0**2*i/10) for i in range(1,90)]
    list_r = [10/i for i in range(1,90)]
    return(list_r,list_correlations)
    
def fractions_list(t_max,proba_negative = 1):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,2))
    current_max = 0
    current_min = 0
    steps = 0
    recordman = 0
    n_max = 0.2*t_max**(1/2)
    ranges = [[0],[0],[0]]
    fractions=[]
    while steps < N and ranges[2][-1] < n_max*discretisation_record:
        steps_x = np.random.normal(0.,np.sqrt(2*D*dt)),np.random.normal(0.,np.sqrt(2*D*diffusion_ratio*dt))
        ## steps_x = levy_stable.rvs(alpha,0,scale = (2*D*dt)**(1/alpha),size = n)
        pos_x[steps+1,0] = pos_x[steps,0] + steps_x[0]
        pos_x[steps+1,1] = pos_x[steps,1] + steps_x[1]
        if pos_x[steps+1,0]<pos_x[steps+1,1]:
            max_pos = pos_x[steps+1,1]
            min_pos = pos_x[steps+1,0]
            if max_pos > current_max:
                beaten =  int((max_pos-current_max)*discretisation_record)
                ranges[1].append(ranges[1][-1]+beaten)
                ranges[0].append(ranges[0][-1])
                ranges[2].append(ranges[2][-1]+beaten)
                fractions.append(ranges[1][-1]/ranges[2][-1])
                current_max = int(discretisation_record*max_pos)/discretisation_record
            if min_pos < current_min:
                beaten =  int((current_min-min_pos)*discretisation_record)
                ranges[0].append(ranges[0][-1]+beaten)
                ranges[1].append(ranges[1][-1])
                ranges[2].append(ranges[2][-1]+beaten)
                fractions.append(ranges[0][-1]/ranges[2][-1])
                current_min = int(discretisation_record*min_pos)/discretisation_record
        if pos_x[steps+1,0]>pos_x[steps+1,1]:
            max_pos = pos_x[steps+1,0]
            min_pos = pos_x[steps+1,1]
            if max_pos > current_max:
                beaten =  int((max_pos-current_max)*discretisation_record)
                ranges[0].append(ranges[0][-1]+beaten)
                ranges[1].append(ranges[1][-1])
                ranges[2].append(ranges[2][-1]+beaten)
                fractions.append(ranges[0][-1]/ranges[2][-1])
                current_max = int(discretisation_record*max_pos)/discretisation_record
            if min_pos < current_min:
                beaten =  int((current_min-min_pos)*discretisation_record)
                ranges[1].append(ranges[1][-1]+beaten)
                ranges[0].append(ranges[0][-1])
                ranges[2].append(ranges[2][-1]+beaten)
                fractions.append(ranges[1][-1]/ranges[2][-1])
                current_min = int(discretisation_record*min_pos)/discretisation_record
        steps+=1
    if ranges[2][-1]>0:
        return(fractions,ranges[2][1:])
    else:
        return([0],[0])
    
def fraction_convergence_check(t_max,proba_negative = 1,it = 1000,split = 20):
    variances = [0 for i in range(split)]
    fractions_l = []
    ranges_tot = []
    for i in range(it):
        fractions, range_tot = fractions_list(t_max,proba_negative=proba_negative)
        fractions_l = np.concatenate((fractions_l,fractions))
        ranges_tot = np.concatenate((ranges_tot,range_tot))
    range_max = max(ranges_tot)
    fraction_binned = [[0.5] for i in range(split)]
    for i in range(len(fractions_l)):
        bin_number = min(int(split*ranges_tot[i]/range_max),split-1)
        fraction_binned[bin_number].append(fractions_l[i])
    for i in range(split):
        variances[i] = np.var(fraction_binned[i])
    dv = [abs(var-3/98) for var in variances]
    return(dv)
    
def exponent_matriochka(t_max,D_ratio=1,proba_negative=1):
    N = int(t_max/dt)
    pos_1,pos_2 = np.zeros(N+1),np.zeros(N+1)
    top_range,bottom_range = 10,0
    steps = 0
    beaten = 0
    pos_2[0] = 5
    while steps < N and beaten == 0:
        increments = np.random.normal(0.,np.sqrt(2*D*dt)),np.random.normal(0.,np.sqrt(2*D*D_ratio*dt))
        pos_1[steps+1] = pos_1[steps]+ increments[0]
        pos_2[steps+1] = pos_2[steps]+ increments[1]
        if pos_1[steps+1] > top_range:
            top_range=pos_1[steps+1]
        if pos_1[steps+1]< bottom_range:
            bottom_range = pos_1[steps+1]
        if pos_2[steps+1] > top_range:
            beaten = 1
        if pos_2[steps+1] < bottom_range:
            x = np.random.random()
            if x < proba_negative:
                beaten = 1
            bottom_range = pos_2[steps+1]
        steps += 1
    return(steps*dt)

def reshuffle_time(t_max,n=2):
    N = int(t_max/dt)
    increments = np.random.normal(0.,np.sqrt(2*D*dt),size = (n,N+1))
    pos = np.cumsum(increments,axis=1)
    pos[0] += np.array([5 for i in range(N+1)])
    shuffle_time = 0
    current_target = 5
    for i, pos_0 in enumerate(pos[0]):
        current_target = max(pos_0,current_target)
        best_candidate = max([pos[j,i] for j in range(1,n)])
        if max(best_candidate,current_target) == best_candidate:
            shuffle_time = i*dt
            break
    if shuffle_time == 0:
        shuffle_time = t_max
    return(shuffle_time)
            

def splitting_matriochka(x,t_max,D_ratio=1,proba_negative=1):
    N = int(t_max/dt)
    pos_1,pos_2,pos_3 = np.zeros(N+1),np.zeros(N+1),np.zeros(N+1)
    top_range,bottom_range = 100,0
    steps = 0
    beaten = 1/2
    pos_2[0] = 100*x
    pos_1[0] = 5
    pos_3[0] = 95
    while steps < N and beaten == 1/2:
        increments = np.random.normal(0.,np.sqrt(2*D*dt)),np.random.normal(0.,np.sqrt(2*D*D_ratio*dt)),np.random.normal(0.,np.sqrt(2*D*dt))
        pos_1[steps+1] = pos_1[steps]+ increments[0]
        pos_2[steps+1] = pos_2[steps]+ increments[1]
        pos_3[steps+1] = pos_3[steps]+ increments[2]
        if pos_1[steps+1] > top_range:
            top_range=pos_1[steps+1]
        if pos_1[steps+1]< bottom_range:
            bottom_range = pos_1[steps+1]
        if pos_3[steps+1] > top_range:
            top_range=pos_3[steps+1]
        if pos_3[steps+1]< bottom_range:
            bottom_range = pos_3[steps+1]
        if pos_2[steps+1] > top_range:
            beaten = 1
        if pos_2[steps+1] < bottom_range:
            x = np.random.random()
            if x < proba_negative:
                beaten = 0
            bottom_range = pos_2[steps+1]
        steps += 1
    if beaten == 1/2:
        return(np.random.randint(0,2))
    else:
        return(beaten)
    
def splitting_expanding(x,t_max,D_ratio = 1,proba_negative=1):
    N = int(t_max/dt)
    pos_1 = np.zeros(N+1)
    top_range,bottom_range = 10,-10
    steps = 0
    beaten = 1/2
    pos_1[0] = -10 + 20*x
    while steps < N and beaten == 1/2:
        increment = np.random.normal(0.,np.sqrt(2*D*D_ratio*dt))
        pos_1[steps+1] = pos_1[steps]+ increment
        if pos_1[steps+1] > top_range:
            beaten = 1
        if pos_1[steps+1]< bottom_range:
            beaten = 0
        else:
            top_range += dt/(2*np.sqrt((steps+1)*dt))
            bottom_range += -dt/(2*np.sqrt((steps+1)*dt))
        steps += 1
    if beaten == 1/2:
        return(np.random.randint(0,2))
    else:
        return(beaten)
    


def excentricity(t_max,n_max,n = 2,proba_negative = 1,alpha=2):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,n))
    range_tot = 0
    current_max = 0
    current_min = 0
    steps = 0
    ranges = np.zeros(n)
    flips = 0
    current_side = 1
    while steps < N and range_tot<n_max*discretisation_record:
        if alpha ==2.:
            steps_x = np.random.normal(0,np.sqrt(2*D*dt),size = n)
            ##steps_x = np.random.normal(drift*dt,np.sqrt(2*D*dt)),np.random.normal(drift*dt,np.sqrt(2*D*diffusion_ratio*dt))
        else:
            steps_x = levy_stable.rvs(alpha,0,scale = (2*D*dt)**(1/alpha),size = n)
        for i in range(n):
            pos_x[steps+1,i] = pos_x[steps,i] + steps_x[i]
        max_pos, min_pos = 0,0
        iM, im = 0,0
        for i,pos in enumerate(pos_x[steps+1]):
            if pos > max_pos:
                iM, max_pos = i,pos
            if pos < min_pos:
                im, min_pos = i,pos
        if max_pos > current_max:
            beaten =  int((max_pos-current_max)*discretisation_record)
            range_tot += beaten
            ranges[iM] += beaten
            if current_side == -1:
                flips += 1
                current_side = 1
            current_max = int(discretisation_record*max_pos)/discretisation_record
            recordman = iM
        if min_pos < current_min:
            beaten =  int(proba_negative*(current_min - min_pos)*discretisation_record)
            range_tot += beaten
            ranges[im]+=beaten
            if current_side == 1:
                flips += 1
                current_side = -1
            current_min = int(discretisation_record*min_pos)/discretisation_record
            recordman = im
        steps+=1
    if range_tot > n_max*discretisation_record:
        final_pos = pos_x[steps,recordman]
        other_pos = np.delete(pos_x[steps],recordman)
        mean_gap = np.mean(pos_x) - final_pos
        gaps = np.array([abs(x - final_pos) for x in other_pos])
        return(flips/np.log(range_tot/discretisation_record),current_max*discretisation_record/range_tot,ranges[0]/range_tot,gaps[0]*discretisation_record/(range_tot))
    else:
        x= np.random.random()
        return(x,x,x,x)

def excentricity_range_controlled(t_max,n_max,n = 2,proba_negative = 1,a=1):
    N = int(t_max/dt)
    pos_x = np.zeros((N+1,n))
    range_tot = 0
    current_max = 0
    current_min = 0
    steps = 0
    ranges = np.zeros(n)
    D = [1/n for i in range(n)]
    while steps < N and range_tot<n_max*discretisation_record:
        steps_x = [np.random.normal(0,np.sqrt(2*dt*D[i])) for i in range(n)]
        for i in range(n):
            pos_x[steps+1,i] = pos_x[steps,i] + steps_x[i]
        max_pos, min_pos = 0,0
        iM, im = 0,0
        for i,pos in enumerate(pos_x[steps+1]):
            if pos > max_pos:
                iM, max_pos = i,pos
            if pos < min_pos:
                im, min_pos = i,pos
        if max_pos > current_max:
            beaten =  int((max_pos-current_max)*discretisation_record)
            range_tot += beaten
            ranges[iM] += beaten
            current_max = int(discretisation_record*max_pos)/discretisation_record
            recordman = iM
        if min_pos < current_min:
            beaten =  int(proba_negative*(current_min - min_pos)*discretisation_record)
            range_tot += beaten
            ranges[im]+=beaten
            current_min = int(discretisation_record*min_pos)/discretisation_record
            recordman = im
        for i in range(n):
            D[i] = ranges[i]**a/np.sum(np.power(ranges,a))
        steps+=1
    if range_tot > n_max*discretisation_record:
        final_pos = pos_x[steps,recordman]
        other_pos = np.delete(pos_x[steps],recordman)
        gap_max = max([abs(final_pos - pos) for pos in other_pos])
        mean_gap = np.mean(other_pos) - final_pos
        gap = other_pos[0] - final_pos
        return(ranges[0]/range_tot,ranges[recordman]/range_tot,abs(gap)*discretisation_record/range_tot)
    else:
        x= np.random.random()
        return(x,x,x)

def time_as_the_record(t_2):
    N = int(t_2/dt)
    pos_x , pos_y = np.zeros(N+1) , np.zeros(N+1)
    pos_x[0], pos_y[0] = 0,0
    t = 0
    max_x = 0
    max_y = 0
    for i in range(N):
        step_x,step_y = np.random.normal(0,np.sqrt(2*D*dt)),np.random.normal(0,np.sqrt(2*D*dt))
        pos_x[i+1] = pos_x[i] + step_x
        pos_y[i+1] = pos_y[i] + step_y
        max_x = max(max_x,pos_x[i+1])
        max_y = max(max_y,pos_y[i+1])
        if max_x>max_y:
            t+=dt  
    return(t/t_2)

def reaching_time_process(x,n=2):
    N = int(x/dx)
    steps = levy.rvs(loc = 0,scale = dx,size = (n,N))
    list_reaching_times = [np.cumsum(step) for step in steps]
    changes = [0]
    sign = -1
    for i in range(N):
        list_reaching_time = [reaching_times[i] for reaching_times in list_reaching_times]
        im = 0
        min_time = max(list_reaching_time)
        for j, t in enumerate(list_reaching_time):
            if t < min_time:
                min_time = t
                im = j
        if im == 1:
            if sign == -1:
                changes.append(changes[-1]+1)
                sign = 1
            else:
                changes.append(changes[-1])
        else:
            if sign == 1:
                changes.append(changes[-1]+1)
                sign = -1
            else:
                changes.append(changes[-1])
    return(changes)

def occupation_time_levy_jump_processes(x):
    N = int(x/dx)
    steps = levy_stable.rvs(1,0,loc = 0,scale = dx,size = 3*N)
    list_reaching_times = np.cumsum(steps)
    occ = 0
    for t in list_reaching_times[N:]:
        if t >0:
            occ += 1
    return(occ/(2*N))

def FCT_levy_jump_processes(alpha,r_max):
    N = 2000
    steps = levy_stable.rvs(alpha,0,loc = 0,scale = dx,size = int((r_max+1)*N))
    list_pos = np.cumsum(steps)
    s = list_pos[N]/abs(list_pos[N])
    for i,p in enumerate(list_pos[N:]):
        if p*s < 0:
            break
    return((i+1)/N)

def reaching_time_process_no_overshoot(x,diffusion_ratio,reset = False):
    N = int(x/dx)
    steps = levy_stable.rvs(1/2,(np.sqrt(diffusion_ratio)-1)/(np.sqrt(diffusion_ratio)+1),loc = 0,scale = dx,size = N)
    list_time_gaps = np.cumsum(steps)
    occupation_times = [0,0]
    recordman = 0
    for i in range(N):
        pos = list_time_gaps[i]
        if pos < 0:
            occupation_times[1]+= 1
            recordman = 1
        else:
            occupation_times[0]+=1
            recordman = 0
    return(occupation_times[0]/N)

def exit_time_interval_cumulative(x,n):
    s = np.sum([np.sin((2*k+1)*np.pi/n)*np.exp(-(2*k+1)**2*np.pi**2*x/(2*n**2))/(2*k+1) for k in range(0,n+1)])
    return(1-4*s/np.pi)
    
def n_simus_pauls_walk(n_steps,n_simus):
    list_pos = np.zeros(n_simus)
    list_hitting_times =[]
    for i in range(2,n_steps+2):
        list_v = np.linspace(0,0.99999999,10000)
        list_F = [exit_time_interval_cumulative(np.tan(np.pi*v/2), i) for v in list_v]
        for j in range(n_simus):
            x1,x2 = np.random.random(),np.random.random()
            k1,k2 = min(np.searchsorted(list_F, x1),len(list_v)-1),min(np.searchsorted(list_F, x2),len(list_v)-1)
            list_pos[j] += np.tan(list_v[k1]*np.pi/2) - np.tan(list_v[k2]*np.pi/2)
    return(list_pos)
    
    
def rho_0(t_2,z):
    N = int(t_2/dt)
    pos_x , pos_y = np.zeros(N+1) , np.zeros(N+1)
    pos_x[0], pos_y[0] = 0,-z
    current_max = 0
    range_2 = 0
    i = 0
    while i < N and range_2  == 0:
        step_x,step_y = np.random.normal(0,np.sqrt(2*D*dt)), np.random.normal(0,np.sqrt(2*D*dt))
        pos_x[i+1] = pos_x[i] + step_x
        pos_y[i+1] = pos_y[i] + step_y
        if pos_x[i+1] > current_max:
            current_max += 1
        if pos_y[i+1] > current_max:
            range_2 += 1
            current_max += 1
        i += 1
    if range_2 == 0:
        return(t_2)
    else:
        return(i*dt)
    
def rho_0_5(t_2):
    N = int(t_2/dt)
    pos_x , pos_y = np.zeros(N+1) , np.zeros(N+1)
    pos_x[0], pos_y[0] = 0,0
    current_max = 1
    range_1 = 0
    range_2 = 0
    for i in range(N):
        step_x,step_y = np.random.normal(0,np.sqrt(2*D*dt)), np.random.normal(0,np.sqrt(2*D*dt))
        pos_x[i+1] = pos_x[i] + step_x
        pos_y[i+1] = pos_y[i] + step_y
        if pos_x[i+1] > current_max:
            range_1 += 1
            current_max += 1
        elif pos_y[i+1] > current_max:
            range_2 += 1
            current_max += 1
    if abs(range_1-range_2) < 2:
        return(1)
    else:
        return(0)


'''
var = fraction_convergence_check(10000,it=30000,split=40)
np.save('range_competition/'+ str(int(time())), var)
'''


def competition_batch(t_max, n_max, n, it, alpha):

    list_x = np.zeros(it)
    correlation_matrix = np.zeros((n_max-1, n_max-1))

    success_count = 0

    for sim in range(it):

        pos = np.zeros(n, dtype=np.int64)
        ranges = np.zeros(n, dtype=np.int64)
        list_labels=[]

        # on remplace le set par un dict-like minimal :
        visited = {0: 1}

        range_tot = 1
        success = False

        for step in range(t_max):

            i = np.random.randint(0, n)
            sign = -1 if np.random.random() < 0.5 else 1
            jump = np.random.zipf(alpha + 1)

            pos[i] += sign * jump
            site = pos[i]

            if site not in visited:
                visited[site] = 1
                ranges[i] += 1
                range_tot += 1
                list_labels.append(i)

                if range_tot == n_max:
                    success = True
                    break

        if success:

            ratio = ranges[0] / range_tot
            list_x[success_count] = ratio

            # accumulation corrélations
            for a in range(n_max-1):
                for b in range(n_max-1):
                    correlation_matrix[a, b] += list_labels[a] * list_labels[b]

            success_count += 1

    return list_x[:success_count], correlation_matrix/it


'''
t1=time()
t = 10**6
it = 10**5
list_x = []
list_z = []
n = int(t**(2/3)/5)

correlation_matrix = np.zeros(((n-1),(n-1)))


for i in range(it):
    p,x,l,z = competition_levy_fast(t,n,alpha=3/2,n=7)
    if p:
        list_x.append(x)
        list_z.append(z)
        
        l_arr = np.asarray([int(i<1) for i in l])
        correlation_matrix += np.outer(l_arr, l_arr)
print(time()-t1)    
##np.save('range_competition/list_z'+ str(int(time())), list_z)
np.save('range_competition/correlationsalpha15n=7'+ str(int(time())), correlation_matrix/it)
'''


t_max=10**7
it= 5*10**5
list_fractions = [[] for i in range(100)]

for i in range(it):
    list_f = autocorrelation_fraction(t_max,proba_negative=1)
    for j in range(100):
        list_fractions[j].append(list_f[j])

np.save('range_competition/fraction_Brownien_1D'+ str(int(time())), list_fractions)



'''
t_max = 10**5
it = 10**5
l0 = np.array(autocorrelation_jump_processes(alpha=1/2,t_max=t_max))

for i in range(it):
    l = np.array(autocorrelation_jump_processes(alpha=1/2,t_max=t_max))
    l0 += l
l0 = l0/(it +1)
np.save('range_competition/' + str(int(time())),l0)
'''



'''
t = 10**5
it = 5*10**4
list_ranges = []
list_gaps = []
list_x_0 = []
list_flips = []

for i in range(it):
    ###x_1 = excentricity(t,np.sqrt(2*t)/10,n=3)
    flips,x_0,x_record,z = excentricity(t,np.sqrt(2*t)/10,n=2,proba_negative = 0.,alpha=2.)
    list_x_0.append(x_0)
    list_gaps.append(z)
    list_ranges.append(x_record)
    list_flips.append(flips)
    
np.save('range_competition/' + str(int(time())),[list_flips,list_ranges,list_gaps])
'''

'''
t = 10**5
it = 6*10**4
list_x_0 = []
list_z = []
list_x_record = []

for i in range(it):
    ###x_1 = excentricity(t,np.sqrt(2*t)/10,n=3)
    x_0,x_record,z = excentricity_range_controlled(t,np.sqrt(2*t)/10,n=2,proba_negative = 1.,a=0.3)
    list_x_0.append(x_0)
    list_z.append(z)
    list_x_record.append(x_record)
    
np.save('range_competition/' + str(int(time())),[list_x_0,list_x_record,list_z])
'''

'''
x_max = 3*3000*dx
it = 3*10**7
list_x = []

for i in range(it):
    x = occupation_time_levy_jump_processes(x_max)
    list_x.append(x)

np.save('range_competition/'+ str(int(time())), list_x)
'''

'''
t_max = 10**5
it= 3*10**3
list_x = np.linspace(0,1,100)
list_probas = np.zeros(100)
for j in range(100):
    for i in range(it):
        list_probas[j] += splitting_matriochka(list_x[j],t_max,D_ratio = 1/2)/it

np.save('range_competition/'+ str(int(time())), list_probas)
'''

'''
t = 10**8
it = 10**5
list_x = []

for i in range(it):
    x,p = range_values(t,n=2,proba_negative=1)
    list_x.append(x)
    
np.save('range_competition/two_sided'+ str(int(time())), list_x)
'''

'''
t = 1000
n = 10**5
list_t = []

for i in range(n):
    list_t.append(time_as_the_record(t))
    
np.save('range_competition/'+ str(int(time())), list_t)
'''


'''
x_max = 50
it = 10**5
list_x_sym = []
list_x_plus = []
list_x_minus = []
for i in range(it):
    x_sym = reaching_time_process_no_overshoot(x_max,1,reset=False)
    x_plus = reaching_time_process_no_overshoot(x_max,3,reset=False)
    x_minus = reaching_time_process_no_overshoot(x_max,1/3,reset=False)
    list_x_sym.append(x_sym)
    list_x_plus.append(x_plus)
    list_x_minus.append(x_minus)

np.save('range_competition/'+ str(int(time())), [list_x_sym,list_x_plus,list_x_minus])
'''



'''
t = 10000
n = 10**4
list_t = []

for i in range(n):
    list_t.append(rho_0(t,0))

np.save('range_competition/'+ str(int(time())), list_t)
'''


'''
list_t = np.logspace(1,3,10)
n = 10**4
list_rho = []

for t in list_t:
    rho = 0
    for n in range(n):
        rho += rho_0_5(t)
    list_rho.append(rho/n)

np.save('range_competition/'+ str(int(time())), list_rho)
'''
