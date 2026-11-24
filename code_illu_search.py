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

def Discrete2DFBM(t,H):
    list_x = davies_harte(t, t, H, generate_gaussian(t))
    list_y = davies_harte(t, t, H, generate_gaussian(t))
    return([int(x) for x in list_x],[int(y) for y in list_y])

t=400
##seed = np.random.randint(10**6)
##print(seed)
np.random.seed(723743)

x1,y1 = Discrete2D(t)
x2,y2 = Discrete2D(t)
sites_visited = set()
discoveries_1,discoveries_2 = [],[]
for i in range(t):
    newpos1 = x1[i],y1[i]
    if newpos1 not in sites_visited:
        sites_visited.add(newpos1)
        discoveries_1.append(newpos1)
    newpos2 = x2[i],y2[i]
    if newpos2 not in sites_visited:
        sites_visited.add(newpos2)
        discoveries_2.append(newpos2)

from matplotlib.colors import ListedColormap

all_cells = discoveries_1+discoveries_2

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
        grid[i, j] = 2  # bleu (prioritaire)

grid[R3-1,R2+4] = 1

# --- Colormap ---
#cmap = ListedColormap(["#D3D3D3", "#B83232", "#346B96","#29271B"])
cmap = ListedColormap(["#DAD6CD", "#B52B2B", "#316690"])

visited1 = set(zip(x1, y1))
visited2 = set(zip(x2, y2))

common_sites = visited1 & visited2

import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

plt.rcParams['figure.dpi']=300
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(2,2, height_ratios=[3,1], hspace=0.4, wspace=-0.35)

ax = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,:])
ax3 = fig.add_subplot(gs[0,1])

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
ax.grid(True,lw=8/size)

from matplotlib.patches import Rectangle

for x, y in common_sites:

    i = offset_y - y
    j = offset_x + x

    ax.add_patch(
        Rectangle(
            (j - 0.5, i - 0.5),  # coin inférieur gauche
            1,
            1,
            facecolor="none",
            edgecolor="k",
            hatch="////",
            linewidth=0.1
        )
    )

# Enlever les labels
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_title(r'\textbf{First-discovery allocation}',fontsize=11,y=1.08)
ax.set_aspect('equal')


Nx = 13  # nombre de cellules horizontales
Ny = 5   # nombre de cellules verticales
gap = -1  # écart entre grilles

alpha = 1.1
beta = 0.35

offset = Nx + alpha * Ny + gap  # décalage horizontal

# =====================
# première grille
# =====================
for x0 in range(Nx + 1):
    y = np.linspace(0, Ny, 100)
    x = x0 * np.ones_like(y)
    x_line = x + alpha * y
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)

for y0 in range(Ny + 1):
    x = np.linspace(0, Nx, 100)
    y = y0 * np.ones_like(x)
    x_line = x + alpha * y
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)

# =====================
# deuxième grille (décalée)
# =====================
for x0 in range(Nx + 1):
    y = np.linspace(0, Ny, 100)
    x = x0 * np.ones_like(y)
    x_line = x + alpha * y + offset
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)

for y0 in range(Ny + 1):
    x = np.linspace(0, Nx, 100)
    y = y0 * np.ones_like(x)
    x_line = x + alpha * y + offset
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)
    
# =====================
# troisième grille (décalée)
# =====================
for x0 in range(Nx + 1):
    y = np.linspace(0, Ny, 100)
    x = x0 * np.ones_like(y)
    x_line = x + alpha * y + 2*offset
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)

for y0 in range(Ny + 1):
    x = np.linspace(0, Nx, 100)
    y = y0 * np.ones_like(x)
    x_line = x + alpha * y + 2*offset
    y_line = beta * y
    ax2.plot(x_line, y_line, color='black', lw=0.5,zorder=0)
    
def get_cell_center(i, j, grid_id):
    """
    Renvoie les coordonnées (x', y') du centre de la cellule (i,j)
    pour la grille 1 ou 2.
    """

    # centre dans la grille non transformée
    x = i - 0.5
    y = j - 0.5

    # décalage pour la 2e grille
    offset = 0
    if grid_id == 2:
        offset = Nx + alpha * Ny + gap
    if grid_id == 3:
        offset = 2*(Nx+alpha*Ny + gap)

    # transformation affine
    x_prime = x + alpha * y + offset
    y_prime = beta * y

    return x_prime, y_prime

list_food_spots_1 = [(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(2, 1),(2, 2),(2, 4),(2, 5),
                     (3, 1),(3, 2),(3, 3),(3, 5),(4, 1),(4, 3),(4, 4),(4, 5),
                     (5, 1),(5, 5),(6, 1),(6, 3),(6, 4),(6, 5),
                     (7, 1),(7, 2),(7, 5),(8, 1),(8, 4),(8, 5),
                     (9, 1),(9, 2),(9, 3),(9, 4),(9, 5),(10, 1),(10, 2),(10, 3),(10, 4),
                     (10, 5),(11, 1),(11, 3),(11, 5),(12, 1),(12, 2),
                     (12, 5),(13, 1),(13, 2),(13, 3),(13, 4),(13, 5)]

list_food_spots_2 = [(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(2, 1),(2, 5),
                     (3, 1),(3, 2),(3, 4),(3, 5),(4, 1),(4, 4),(4, 5),
                     (5, 1),(5, 3),(5, 4),(5, 5),(6, 1),(6, 4),(6, 5),
                     (7, 1),(7, 5),(8, 1),(8, 2),(8, 3),(8, 5),
                     (9, 1),(9, 2),(9, 5),(10, 1),(10, 2),(10, 3),
                     (10, 5),(11, 1),(11, 2),(11, 3),(11, 5),(12, 1),(12, 2),
                     (12, 3),(12, 4),(12, 5),(13, 1),(13, 2),(13, 3),(13, 4),(13, 5)]

list_food_spots_3 = [(1,1),(1,3),(1,5),(2,3),(2,5),(3,1),(4,4),(6,4),(7,4),(7,5),(8,1),
                     (8,5),(9,1),(9,5),(10,5),(11,2),(12,1),(12,4),(12,5),(13,4)]

list_blocked_spots_3 = [(i,j) for i in range(1,Nx+1) for j in range(1,Ny+1) if np.random.random() < 0.55]



plt.rcParams['text.usetex'] = False
img_feuille = mpimg.imread('/users/invites/plaud/Documents/articles/article competitive foraging/feuille.png') #image de feuille
img_feuille_rot = ndimage.rotate(img_feuille, angle=-90, reshape=True)
imagebox_feuille = OffsetImage(img_feuille_rot, zoom=0.00066)
for cell in list_food_spots_1:
    a = AnnotationBbox(imagebox_feuille, get_cell_center(cell[0],cell[1],2), frameon=False)
    ax2.add_artist(a)

for cell in list_food_spots_2:
    a = AnnotationBbox(imagebox_feuille, get_cell_center(cell[0],cell[1],1), frameon=False)
    ax2.add_artist(a)

for cell in list_food_spots_3:
    a = AnnotationBbox(imagebox_feuille, get_cell_center(cell[0],cell[1],3), frameon=False)
    ax2.add_artist(a)

plt.rcParams['text.usetex'] = True


vx = np.array([1.0, 0.0])
vy = np.array([alpha, beta])

from matplotlib.patches import Polygon

for cell in list_blocked_spots_3:
    center = get_cell_center(cell[0],cell[1],3) 
    verts = [center - 0.5*vx - 0.5*vy,center + 0.5*vx - 0.5*vy,center + 0.5*vx + 0.5*vy,center - 0.5*vx + 0.5*vy]
    poly = Polygon(verts,facecolor='black',edgecolor='none',alpha=0.7)
    ax2.add_patch(poly)
    


ax2.scatter(get_cell_center(3,4,2)[0],get_cell_center(3,4,2)[1],color='#B83232',s=30,zorder=10)
ax2.scatter(get_cell_center(12,3,2)[0],get_cell_center(12,3,2)[1],color='#346B96',s=30,zorder=10)


style = "Simple, tail_width=0.2, head_width=5, head_length=5"
kw = dict(arrowstyle=style,zorder=10)
ax2.add_patch(patches.FancyArrowPatch(get_cell_center(3,4,2),get_cell_center(9,4,2),
                                      connectionstyle="arc3,rad=-0.3",color = '#B83232', **kw))
ax2.add_patch(patches.FancyArrowPatch(get_cell_center(12,3,2),get_cell_center(6,2,2),
                                      connectionstyle="arc3,rad=-0.3",color='#346B96', **kw))
ax2.text(get_cell_center(1,Nx//2+1,1)[0],-0.4,r'short-range hopping',color='black',fontsize=8,ha='center')
ax2.text(get_cell_center(1,Nx//2+1,1)[0],-0.7,r'\textbf{ordinary diffusion}',color='black',fontsize=10,ha='center')

ax2.scatter(get_cell_center(3,3,1)[0],get_cell_center(3,3,1)[1],color='#B83232',s=30,zorder=10)
ax2.scatter(get_cell_center(9,3,1)[0],get_cell_center(9,3,1)[1],color='#346B96',s=30,zorder=10)

ax2.add_patch(patches.FancyArrowPatch(get_cell_center(3,3,1),get_cell_center(4,3,1),
                                      connectionstyle="arc3,rad=-3",color = '#B83232', **kw))
ax2.add_patch(patches.FancyArrowPatch(get_cell_center(9,3,1),get_cell_center(9,2,1),
                                      connectionstyle="arc3,rad=-2",color='#346B96', **kw))

ax2.text(get_cell_center(1,Nx//2 +1,2)[0],-0.4,r'long-range jumps',color='black',fontsize=8,ha='center')
ax2.text(get_cell_center(1,Nx//2+1,2)[0],-0.7,r'\textbf{superdiffusion}',color='black',fontsize=10,ha='center')

ax2.add_patch(patches.FancyArrowPatch(get_cell_center(6,2,3),get_cell_center(6,3,3),
                                      connectionstyle="arc3,rad=-1.5",color = '#B83232', **kw))
ax2.add_patch(patches.FancyArrowPatch(get_cell_center(11,3,3),get_cell_center(11,2,3),
                                      connectionstyle="arc3,rad=-2",color='#346B96', **kw))
ax2.scatter(get_cell_center(6,2,3)[0],get_cell_center(6,2,3)[1],color= '#B83232',s=30,zorder=10)
ax2.scatter(get_cell_center(11,3,3)[0],get_cell_center(11,3,3)[1],color= '#346B96',s=30,zorder=10)

ax2.text(get_cell_center(1,Nx//2+1,3)[0],-0.4,r'disordered/crowded environments',fontsize=8,ha='center')
ax2.text(get_cell_center(1,Nx//2+1,3)[0],-0.7,r'\textbf{subdiffusion}',color='black',fontsize=10,ha='center')

ax2.set_aspect(3.5)
ax2.axis('off')
ax2.set_title(r'\textbf{General transport classes}',fontsize=11,y=1.14)

graph_width = 1
graph_spacing_x=1.5
graph_spacing_y=2

n1 = np.linspace(0,graph_width,100)
ax3.plot(n1,80*n1**3*(graph_width-n1)**3,lw=0.8,color='black')
ax3.plot(n1,80*n1**3*(graph_width-n1)**3+graph_spacing_y,lw=0.8,color='black')
ax3.annotate("", xytext=(-0.05,0), xy=(graph_width+0.05,0),
            arrowprops=dict(arrowstyle="-|>",color='black',lw=0.5),color='black')
ax3.annotate("", xytext=(-0.05,graph_spacing_y), xy=(graph_width+0.05,graph_spacing_y),
            arrowprops=dict(arrowstyle="-|>",color='black',lw=0.5),color='black')
ax3.vlines((0.5*graph_width,graph_spacing_x + 1.5*graph_width),0,0.1,color='black',lw=0.5)
ax3.vlines((0.5*graph_width,graph_spacing_x + 1.5*graph_width),graph_spacing_y,graph_spacing_y+0.1,color='black',lw=0.5)

n2 = np.linspace(graph_width+graph_spacing_x,2*graph_width+graph_spacing_x,100)
ax3.plot(n2,26000012**2*(n2-graph_width-graph_spacing_x)**24*(graph_spacing_x + 2*graph_width-n2)**24+graph_spacing_y,lw=0.8,color='black')
ax3.plot(n2,80*(n2-graph_width -graph_spacing_x)**3*(2*graph_width+graph_spacing_x-n2)**3,lw=0.8,color='black')
ax3.annotate("", xytext=(graph_width+graph_spacing_x-0.05,0), xy=(graph_spacing_x+2*graph_width+0.05,0),
            arrowprops=dict(arrowstyle="-|>",color='black',lw=0.5),color='black')
ax3.annotate("", xytext=(graph_width+graph_spacing_x-0.05,graph_spacing_y), xy=(graph_spacing_x+2*graph_width+0.05,graph_spacing_y),
            arrowprops=dict(arrowstyle="-|>",color='black',lw=0.5),color='black')

ax3.annotate(r"$x$", xytext=(graph_width-0.08,-0.25),xy=(0,0),color='black',fontsize=8)
ax3.annotate(r"$x$", xytext=(2*graph_width+graph_spacing_x-0.08,-0.25),xy=(0,0),color='black',fontsize=8)
ax3.annotate(r"$x$", xytext=(graph_width-0.08,graph_spacing_y-0.25),xy=(0,0),color='black',fontsize=8)
ax3.annotate(r"$x$", xytext=(2*graph_width+graph_spacing_x-0.08,graph_spacing_y-0.25),xy=(0,0),color='black',fontsize=8)

ax3.annotate(r"$\langle X_n \rangle$", xytext=(graph_width/2,-0.25),xy=(0,0),color='black',fontsize=8,ha='center')
ax3.annotate(r"$\langle X_n \rangle$", xytext=(1.5*graph_width+graph_spacing_x,-0.25),xy=(0,0),color='black',fontsize=8,ha='center')
ax3.annotate(r"$\langle X_n \rangle$", xytext=(graph_width/2,graph_spacing_y-0.25),xy=(0,0),color='black',fontsize=8,ha='center')
ax3.annotate(r"$\langle X_n \rangle$", xytext=(1.5*graph_width+graph_spacing_x,graph_spacing_y-0.25),xy=(0,0),color='black',fontsize=8,ha='center')

ax3.annotate(r'$X_n = \frac{\text{number of red sites among first }n \text{ discoveries}}{n}$',
             xytext=(graph_width+graph_spacing_x/2,graph_spacing_y + 2.5),xy=(0,0),color='black',ha='center',fontsize=14)
ax3.annotate(r'\begin{center} Concentration towards the \\ mean for transient walks \end{center}',
             xytext=(graph_width+graph_spacing_x/2,graph_spacing_y + 1),xy=(0,0),color='black',ha='center',fontsize=8)
ax3.annotate(r'\begin{center} Persistent randomness \\ for recurrent walks \end{center}',
             xytext=(graph_width+graph_spacing_x/2,1),xy=(0,0),color='black',ha='center',fontsize=8)
ax3.annotate("", xytext=(graph_width+0.2,graph_spacing_y-0.45), xy=(graph_spacing_x+graph_width-0.2,graph_spacing_y-0.45),
            arrowprops=dict(arrowstyle="-|>",color='black',lw=1),color='black')
ax3.annotate(r'as $n$ increases', xytext = (graph_width+graph_spacing_x/2,graph_spacing_y -0.35),xy=(0,0),color='black',fontsize=10,ha='center')
ax3.set_title(r'\textbf{The self-averaging transition}',fontsize=11,x=0.43,y=1.08)

ax3.set_aspect(0.5)
ax3.set_ylim(-0.1,graph_spacing_y +3)
ax3.set_xlim(0,2*graph_width+graph_spacing_x+0.5)
ax3.axis('off')

ytop = (ax3.title.get_window_extent().y0 + ax3.title.get_window_extent().y1)/3000
ylow = (ax2.title.get_window_extent().y0 + ax2.title.get_window_extent().y1)/3000

fig.text(0.25, ytop, r'\textbf{(a)}', fontsize=12,va='center')
fig.text(0.5, ytop, r'\textbf{(b)}', fontsize=12, va='center')
fig.text(0.25, ylow, r'\textbf{(c)}', fontsize=12)

fig.savefig('/users/invites/plaud/Documents/articles/article competitive foraging/illu_competitive_search.png',transparent=True)
plt.show()