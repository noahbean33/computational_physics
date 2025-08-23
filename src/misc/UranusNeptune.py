""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work.""""

# UranusNeptune.py: Orbits of Neptune & Uranus

from visual.graph import *
scene=display(width=600,height=600,  title= 'White Neptune & Black Uranus', range=40)
sun = sphere(pos=(0,0,0), radius=2, color=color.yellow)
escenau=gdisplay(x=600,width=400,height=400, title='Pertubation of Uranus Angular Position')
graphu=gcurve(color=color.cyan)

escenan=gdisplay(x=800,y=400,width=400,height=400)
graphn=gcurve(color=color.white)
rfactor=1.8e-9
G=4*pi*pi         # in units T in years, R AU, Msun=1
mu=4.366244e-5    # mass Uranus in solar masses
M=1.0             # mass Sun
mn=5.151389e-5    # Neptune mass in solar masses
#du=19.19126393       # distance Uranus Sun in AU
du=19.1914        # distance Uranus Sun in AU
dn=30.0611        # distance Neptune sun in AU
#omeur=0.07479    # Uranus ang vel ,units 1/yr
#omeur=0.07474246
#omeur=0.074790031  #Uranus ang vel 1/yr
#omenp=0.0381284  # Neptune ang vel,(1/yr)

#Tur=84.01        # Uranus Orbital Period yr
Tur=84.0110       # Uranus Orbital Period yr
Tnp=164.7901      # Neptune Orbital Period yr
omeur=2*pi/Tur    # Uranus angular velocity (2pi/T)
omennp=2*pi/Tnp   # Neptune angular velocity
omreal=omeur
#dp=d*rfactor
urvel=2*pi*du/Tur    # Uranus orbital velocity UA/yr
npvel=2*pi*dn/Tnp    # Neptune orbital velocity UA/yr
# 1 Uranus at lon 2 gr 16 min sep 1821
radur=(205.64)*pi/180.  # to radians in 1690 -wrt x-axis
urx=du*cos(radur)       #init x- pos ur. in 1690
ury=du*sin(radur)       #init y-pos ur in 1690
urvelx=urvel*sin(radur)
urvely=-urvel*cos(radur)
# 1690 Neptune at long. 
radnp=(288.38)*pi/180. # 1690 radians neptune angular pos. wrt x-axis
uranus=sphere(pos=(urx,ury,0), radius=0.5,color=(.88,1,1), make_trail=True)
urpert=sphere(pos=(urx,ury,0), radius=0.5,color=(.88,1,1), make_trail=True)
fnu=arrow(pos=uranus.pos,color=color.orange,axis=vector(0,4,0))#on uranus deu to N
npx=dn*cos(radnp)       #init coord x neptune 1690
npy=dn*sin(radnp)       #           y
npvelx=npvel*sin(radnp)
npvely=-npvel*cos(radnp)
#fusun=arrow(pos=uranus.pos,color=color.orange,axis=vector(0,-4,0))# U  duej to s
neptune=sphere(pos=(npx,npy,0), radius=0.4,color=color.cyan, make_trail=True)
fun=arrow(pos=neptune.pos,color=color.orange,axis=vector(0,-4,0))#N due to s-ur
nppert=sphere(pos=(npx,npy,0), radius=0.4,color=color.white, make_trail=True)
velour=vector(urvelx,urvely,0)   #initial vector velocity Uranus
velnp=vector(npvelx,npvely,0)    #initial vector velocity Neptune
dt=0.5                    # time increment in terrestrial year
r=vector(urx,ury,0)           # initial position Uranus wrt Sun
rnp=vector(npx,npy,0)         # initial position Neptune wrt Sun
veltot=velour

veltotnp=velnp
rtot=r
rtotnp=rnp

def ftotal(r,rnp,i):   # i==1 Uranus  i==2 Neptune
    Fus= -G*M*mu*r/(du**3)  # Force sun over URANUS
    Fns=-G*M*mn*rnp/(dn**3) # Force Sun over NEPTUNE
    dnu=mag(rnp-r)          # distance Neptune-Uranus
    Fnu=-G*mu*mn*(rnp-r)/(dnu**3) # force NEPTUNE on URANUS
    Fun=-Fnu                    # force uranus over Neptune
    Ftotur=Fus+Fnu                # total for on Uranus (sun +Neptune)
    Ftotnp=Fns+Fun                # On Neptune F sun +F urn
    if i==1:
       return Ftotur
    else:
        return Ftotnp
    
def rkn(r,veltot,rnp,m,i):    # on Neptune
    k1v=ftotal(r,rnp,i)/m
    k1r=veltot
    k2v=ftotal(r,rnp+0.5*k1r*dt,i)/m
    k2r=veltot+0.5*k2v*dt
    k3v=ftotal(r,rnp+0.5*k2r*dt,i)/m
    k3r=veltot+0.5*k3v*dt
    k4v=ftotal(r,rnp+k3r*dt,i)/m
    k4r=veltot+k4v*dt
    veltot=veltot+(k1v+2*k2v+2*k3v+k4v)*dt/6.0
    rnp=rnp+(k1r+2*k2r+2*k3r+k4r)*dt/6.0
    return r,veltot

def rk(r,veltot,rnp,m,i):   # on Uranus
    k1v=ftotal(r,rnp,i)/m
    
    k1r=veltot
    k2v=ftotal(r+0.5*k1r*dt,rnp,i)/m
    k2r=veltot+0.5*k2v*dt
    k3v=ftotal(r+0.5*k2r*dt,rnp,i)/m
    k3r=veltot+0.5*k3v*dt
    k4v=ftotal(r+k3r*dt,rnp,i)/m
    k4r=veltot+k4v*dt
    veltot=veltot+(k1v+2*k2v+2*k3v+k4v)*dt/6.0
    r=r+(k1r+2*k2r+2*k3r+k4r)*dt/6.0
    return r,veltot

for i in arange(0,320):# estaba 1240
    rate(10)
    rnewu,velnewu=rk(r,velour,rnp,mu,1) #uranus
    
    rnewn,velnewn=rkn(rnp,velnp,r,mn,2) #neptune
    r=rnewu                      # uranus position
    velour=velnewu               # uranus velocity
    du=mag(r)
    omeur=mag(velour)/du         # new abgykar velocity of uranus
    degr=205.64*pi/180- omeur*i*dt # angular position uranus
    rnp=rnewn                   # neptune pos
    velnp=velnewn                # neptune pos
    dn=mag(rnp)
    omenp=mag(velnp)/dn
    radnp=radnp-dt*omenp           # radians neptune
    npx=dn*cos(radnp) 
    npy=dn*sin(radnp)
    rnp=vector(npx,npy,0)       # neptune position   
    deltaomgs=-omeur+omreal
    #if i%20==0:
    graphu.plot(pos=(i,deltaomgs*180/pi*3600))
    urpert.pos=r
    fnu.pos=urpert.pos            # position of arrow on uranus
    dnu=mag(rnp-r)          # distance Neptune-Uranus
    fnu.axis=75*norm(rnp-r)/dnu   # axes  the arrow over uranus
    neptune.pos=rnp         # radiovector Neptune
    fun.pos=neptune.pos
    fun.axis=-fnu.axis            # arrow on neptune
