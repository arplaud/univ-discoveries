import matplotlib.pyplot as plt
import numpy as np
from time import time
from collections import defaultdict
import sys

# Davies Harte method

# Generate the initial trajectory


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


dt = 1


def simu_2D_fBm(N,list_indices, H):
    T = int(N**max(1,1/(2*H))*np.log(N))//3
    fbm1 = davies_harte(T, T, H, generate_gaussian(T))
    fbm2 = davies_harte(T, T, H, generate_gaussian(T))
    fbm3 = davies_harte(T, T, H, generate_gaussian(T))
    fbm4 = davies_harte(T, T, H, generate_gaussian(T))
    fbm2D1 = np.array([(fbm1[i], fbm2[i]) for i in range(T)])
    fbm2D2 = np.array([(fbm3[i], fbm4[i]) for i in range(T)])
    pos = set()
    ranges = [0, 0]
    M = 0
    list_frac = defaultdict(list)
    list_labels = defaultdict(int)
    tf=T
    for i in range(0, T):
        pos1 = int(fbm2D1[i][0]), int(fbm2D1[i][1])
        pos2 = int(fbm2D2[i][0]), int(fbm2D2[i][1])
        if pos1 not in pos:
            pos.add(pos1)
            ranges[0] += 1
            M += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
                list_labels[str(M)] = 1
        if pos2 not in pos:
            pos.add(pos2)
            ranges[1] += 1
            M += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
                list_labels[str(M)] = 0
        if M >= N:
            tf = i
            break
    return (list_frac,list_labels,tf/T)

def simu_3D_fBm(N,list_indices, H):
    T = int(N**max(1,1/(3*H))*np.log(N))//10
    fbm1 = davies_harte(T, T, H, generate_gaussian(T))
    fbm2 = davies_harte(T, T, H, generate_gaussian(T))
    fbm5 = davies_harte(T, T, H, generate_gaussian(T))
    fbm3 = davies_harte(T, T, H, generate_gaussian(T))
    fbm4 = davies_harte(T, T, H, generate_gaussian(T))
    fbm6 = davies_harte(T, T, H, generate_gaussian(T))
    fbm3D1 = np.array([(fbm1[i], fbm2[i],fbm5[i]) for i in range(T)])
    fbm3D2 = np.array([(fbm3[i], fbm4[i],fbm6[i]) for i in range(T)])
    pos = set()
    ranges = [0, 0]
    M = 0
    list_frac = defaultdict(list)
    tf=T
    for i in range(0, T):
        pos1 = int(fbm3D1[i][0]), int(fbm3D1[i][1]), int(fbm3D1[i][2])
        pos2 = int(fbm3D2[i][0]), int(fbm3D2[i][1]), int(fbm3D2[i][2])
        if pos1 not in pos:
            pos.add(pos1)
            ranges[0] += 1
            M += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
        if pos2 not in pos:
            pos.add(pos2)
            ranges[1] += 1
            M += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
        if M >= N:
            tf = i
            break
    return (list_frac,tf/T)

def simu_2DBM(N,list_indices):
    T = int(N*np.log(N))
    steps=0
    M=1
    visited=set()
    list_frac=defaultdict()
    visited.add((0,0))
    pos1=(0,0)
    pos2=(0,0)
    ranges=[0,0]
    moves = [(0,1),(0,-1),(1,0),(-1,0)]
    while M<N and steps < T:
        i1 = np.random.randint(0,4)
        i2 = np.random.randint(0,4)
        pos1 = pos1[0] + moves[i1][0], pos1[1] + moves[i1][1]
        pos2 = pos2[0] + moves[i2][0], pos2[1] + moves[i2][1]
        if pos1 not in visited:
            visited.add(pos1)
            ranges[0]+=1
            M+=1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
        if pos2 not in visited:
            visited.add(pos2)
            ranges[1]+=1
            M+=1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
    return(list_frac)

def simu_3DBM(N,list_indices,theta=0):
    ##no food on odd planes
    T = int(N*np.log(N))
    steps=0
    M=1
    visited=set()
    list_frac=defaultdict()
    visited.add((0,0,0))
    pos1=(0,0,0)
    pos2=(0,0,0)
    ranges=[0,0]
    moves = [(0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)]
    while M<N and steps < T:
        i1 = np.random.randint(0,6)
        i2 = np.random.randint(0,6)
        pos1 = pos1[0] + moves[i1][0], pos1[1] + moves[i1][1], pos1[2] + moves[i1][2]
        pos2 = pos2[0] + moves[i2][0], pos2[1] + moves[i2][1], pos2[2] + moves[i2][2]
        ##test_pos_1 = (pos1[2] > 0) and (pos1[0]**2 + pos1[1]**2 <= abs(pos1[2])**(3/2)*np.tan(theta)**2)
        ##test_pos_2 = (pos2[2] > 0) and (pos2[0]**2 + pos2[1]**2 <= abs(pos2[2])**(3/2)*np.tan(theta)**2)
        test_pos_1 = (pos1[2]%2 == 0)
        test_pos_2 = (pos2[2]%2 == 0)
        if (pos1 not in visited) and (not test_pos_1):
            visited.add(pos1)
            ranges[0]+=1
            M+=1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
        if (pos2 not in visited) and (not test_pos_2):
            visited.add(pos2)
            ranges[1]+=1
            M+=1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
    return(list_frac)


def move(pt, lisdir):
    if pt == (0, 0):
        return lisdir[np.random.randint(1, 3)]
    else:
        list_pos = []
        for d in lisdir:
            c = (pt[0]-d[0], pt[1]-d[1])
            c = c[0] & c[1]
            if c == 0:
                list_pos.append((-d[0], -d[1]))
        c = list_pos[np.random.randint(len(list_pos))]
        lisdir2 = lisdir[:]
        while (-c[0], -c[1]) in lisdir2:
            lisdir2.remove((-c[0], -c[1]))
        c2 = lisdir2[np.random.randint(len(lisdir2))]
        return (pt[0]+c[0]+c2[0], pt[1]+c[1]+c2[1])


def simu_sierpinski(N, lisdir,list_indices,n=2):
    pos1 = (0, 0)
    pos2 = (0, 0)
    pos = set()
    pos.add(pos1)
    ranges = [0,0]
    list_frac = defaultdict(list)
    M = 1
    list_frac['1'] = 1/2
    while M < N:
        pos1 = move(pos1, lisdir)
        pos2 = move(pos2, lisdir)
        if pos1 not in pos:
            pos.add(pos1)
            M += 1
            ranges[0] += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
        if pos2 not in pos:
            pos.add(pos2)
            M += 1
            ranges[1] += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[0]+ranges[1])
    return (list_frac)


def move_comb(pos):
    x = np.random.randint(1, 5)
    if pos[1] == 0:
        pos[x % 2] += int((x//2 - 1.5 + x % 2)*2)
    else:
        pos[1] += int((x % 2 - 0.5)*2)
    return (pos)


def simu_1D_Levy(N, list_indices,alpha=1, n=2,betas=None):
    if betas == None:
        betas = np.zeros(n)
    t_max = int(10*N**(max(1, min(alpha,2)))*np.log(N+2))
    sites_visited_tot = set()
    sites_visited_tot.add(0)
    range_tot = 0
    ranges = np.zeros(n)
    list_frac = defaultdict(list)
    pos = n*[0]
    M = 1
    steps = 0
    while len(sites_visited_tot) < N and steps < t_max:
        for j in range(n):
            sign = 2*np.random.randint(0, 2) - 1
            jump = int((1+betas[j])*np.random.zipf(alpha+1))
            pos[j] += sign*jump
        for j in range(n):
            newpos = pos[j]
            if newpos not in sites_visited_tot:
                sites_visited_tot.add(newpos)
                range_tot += 1
                ranges[j] += 1
                M+=1
                if M in list_indices:
                    list_frac[str(M)] = ranges[0]/range_tot
        steps += 1
    return (list_frac)

'''
index=int(sys.argv[1]) # When you want to do simus on the cluster, to specify which job uses which percolation cluster
K=index
#K=1

vois=defaultdict(list)
filename = '/users/invites/plaud/Documents/perc_voisins/largest_cluster_'+str(K)+'.dat' # Take a single one of the 31 we have
#the file contains all the combination of neighbours 
Tm = np.loadtxt(filename)[:]
Tm = Tm.astype(int)
for i in range(len(Tm)):
  vois[Tm[i][0]].append(Tm[i][1])
  vois[Tm[i][1]].append(Tm[i][0])
vois=dict(vois)


list_pos=np.int64(np.array(list(vois.keys())))



def simu_percolation(N,vois,list_indices,n=2):
    a=list_pos[np.random.randint(len(list_pos))]
    c1,c2 = a,a
    pos=set()
    ranges = [0,0]
    list_frac = defaultdict(list)
    list_labels = defaultdict(int)
    pos.add(c1)
    M=1
    list_frac['1'] = 0.5
    while M<N:
       c1=vois[np.int64(c1)][np.random.randint(len(vois[c1]))]
       c2=vois[np.int64(c2)][np.random.randint(len(vois[c2]))]
       if not (c1 in pos):
            pos.add(c1)
            M+=1
            ranges[0] += 1
            if M in list_indices:
                list_frac[str(M)] = ranges[0]/(ranges[1]+ranges[0])
                list_labels[str(M)] = 0
       if not (c2 in pos):
           pos.add(c2)
           M+=1
           ranges[1] += 1
           if M in list_indices:
               list_frac[str(M)] = ranges[0]/(ranges[1]+ranges[0])
               list_labels[str(M)] = 1
    return list_frac,list_labels

t1=time()
it = 5*10**4
N = 2000
list_indices = range(2,N)
list_frac = defaultdict(list)

correlation_matrix = np.zeros(((N-2),(N-2)))

for i in range(it):
    f,l = simu_percolation(N,vois,list_indices,n=2)
    for i in list_indices:
        list_frac[str(i)].append(f[str(i)])
    l = np.asarray([l[str(i)] for i in list_indices])
    correlation_matrix += np.outer(l,l)

list_frac = np.array([list_frac[str(i)] for i in list_indices])
print(time()-t1)
np.save('/users/invites/plaud/Documents/range_competition/perco_corr' +
        str(int(time())), correlation_matrix/it)

'''
t1=time()
it = 10**3
N = 10**7
list_indices = np.asarray(np.logspace(1,int(np.log10(N)),50),dtype=np.int64)

list_frac = defaultdict(list)

for i in range(it):
    f= simu_3DBM(N,list_indices,theta=0)
    for j in list_indices:
        list_frac[str(j)].append(f[str(j)])

print(time()-t1)
list_frac = np.array([list_frac[str(i)] for i in list_indices])
np.save('/users/invites/plaud/Documents/range_competition/fraction3DBModd' +
        str(int(time())), list_frac)


#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.patches as patches

# -----------------------------
# Paramètres
# -----------------------------
L = 200
p = 0.5
seed = np.random.randint(10000)

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
# Numérotation des sommets
# -----------------------------
node_id = {node: i for i, node in enumerate(sorted(largest_cluster))}

# -----------------------------
# Construction de la liste des arêtes
# -----------------------------
edges = []

for x, y in sorted(largest_cluster):

    i = node_id[(x, y)]

    # voisin de droite
    if x < L - 1 and horizontal[y, x] and (x + 1, y) in node_id:
        j = node_id[(x + 1, y)]
        edges.append((i, j))

    # voisin du haut
    if y < L - 1 and vertical[y, x] and (x, y + 1) in node_id:
        j = node_id[(x, y + 1)]
        edges.append((i, j))

# ordre lexicographique
edges.sort()


# -----------------------------
# Sauvegarde
# -----------------------------
if len(largest_cluster) > 18000 and len(largest_cluster) < 22000:
    with open("/users/invites/plaud/Documents/perc_voisins/largest_cluster_20.dat", "w") as f:
        for i, j in edges:
            f.write(f"{i}\t{j}\n")

