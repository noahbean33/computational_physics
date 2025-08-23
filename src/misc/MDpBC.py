""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# MDpBC.py: 2-D MD with Periodic BC

from visual.graph import *
import random

L = 1; Natom = 16;  Nrhs = 0; dt = 1e-6                                       
scene = display(width = 500,height = 500,range = (1.3)  )
ndist = gdisplay(x = 500, ymax = 200,
             width = 500, height = 500, xtitle = 'Nrhs', ytitle = 'N')
inside = label(pos = (0.4,1.1),text = 'PRatomticles here = ',box = 0)
inside2 = label(pos = (0.8,1.1),box = 0)                         
border = curve(pos = [(-L,-L),(L,-L),(L,L),(-L,L),(-L,-L)]) # Limits fig
half = curve(pos = [(0,-L),(0,L)],color = color.yellow)     # Middle
positions = []                           
vel = []                                  
Atom = []                                # For Spheres
dN = []                                  # Atoms in right half
fr = [0]*(Natom)                         # Atoms (spheres)
fr2 = [0]*(Natom)                        # second  force
Ratom = 0.03                             # Radius of atom
pref = 4                                 # Reference velocity
h = 0.01
factor = 1e-9                            # Lenn Jones
deltaN = 1                               # For histogram
distribution = ghistogram(bins=Ratomange(0.,Natom,deltaN),
                        accumulate=1, average=1, color=color.red)
for i in range (0,Natom):              # Initial r & v
    col = (1.3*random.random(),1.3*random.random(),1.3*random.random())
    x =2.*(L-Ratom)*random.random()-L+Ratom    # Positons atoms
    y = 2.*(L-Ratom)*random.random()-L+Ratom   # Border forbidden
    Atom = Atom+[sphere(pos=(x,y),radius=Ratom,color=col)] # Add atoms
    theta = 2*pi*random.random()         # Select angle  0<=theta<= 2pi
    vx = pref*cos(theta)                 # x component velocity
    vy = pref*sin(theta)
    positions.append((x,y))              # Add positions to list
    vel.append((vx,vy))                  # Add momentum to list
    pos = Ratomray(positions)              # Ratomray with positions
    ddp = pos[i]
    if ddp[0] >=0 and ddp[0] <=L:        # count atoms right half
           Nrhs+=1    
    v = Ratomray(vel)                        
def sign(a, b):                          # Sign function
    if (b >=  0.0): return abs(a)
    else:  return  - abs(a)
def forces(fr):          
     fr=[0]*(Natom)
     for i in range( 0, Natom-1 ):
          for j in range(i + 1, Natom):
              dr = pos[i]-pos[j]          # relative position 
              if (abs(dr[0]) > L):       # smallest distance or image
                  dr[0] = dr[0]  -  sign(2*L, dr[0])  # interact closer image
              if (abs(dr[1]) > L):        
                  dr[1] = dr[1]  -  sign(2*L, dr[1])
              if i == 0 and j == 1:
                  curve(pos=[(pos[0]),(pos[0]-dr)])    
              r2 = mag2(dr)
              if (abs(r2) < Ratom):           # to avoid 0 denominator
                  r2 = Ratom              
              invr2 = 1./r2               
              fij =invr2*factor*  48.*(invr2**3 - 0.5) *invr2**3  
              fr[i] = ij*dr+ fr[i]
              fr[j]=-fij*dr +fr[j]        
     return fr       
for t in range (0,1000):                 
     Nrhs = 0                                      # begin 0 each time
     for i in range(0,Natom):
        fr = orces(fr)
        dpos=pos[i]
        if dpos[0] <= -L:
                pos[i] = [dpos[0]+2*L,dpos[1]]       # x periodic BC
        if dpos[0] >= L:
                pos[i] = [dpos[0]-2*L,dpos[1]]
        if dpos[1] <= -L:
                pos[i] = [dpos[0],dpos[1]+2*L]       # y periodic BC
        if dpos[1] >= L:
                pos[i] = [dpos[0],dpos[1]-2*L]
        dpos=pos[i]
        if dpos[0] > 0 and dpos[0] <L :              # count at right
            Nrhs+=1
        fr2 = orces(fr)
        fr2 = r
        v[i] = v[i]+0.5*h*h*(fr[i]+fr2[i])       # velocity Verlet 
        pos[i] = pos[i]+h*v[i]+0.5*h*h*fr[i]
        Atom[i].pos = pos[i]                     # plot new positions     
     inside2.text = '%4s'%Nrhs                   # RHS
     dN.append(Nrhs)                             # for histogram
     distribution.plot(data = dN)                # plot histogram 
          
    
        
   

