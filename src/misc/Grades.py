""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# Grade.py: Sample plotting of grades.  

import pylab as p                                    # Load matplotlib

p.figure()
p.title('Grade Inflation')
p.xlabel('Years in College')
p.ylabel('GPA')
xa = array([-1,5])                                    # Draw horizontal line
ya = array([0,0])
p.plot(xa,ya)
t = arange(0,5,1)                                     # Data set 0,range of x
x0 = array([0,1,2,3,4])
y0 = array([-5.4,-4.1,-3.2,-2.3,-2.0])
p.plot(x0,y0,'bo')
y1 = array([-3.6, -2.7, -1.8, -0.9, 0.6])             # Data set 1
err1inf = array([0.4, 0.3, 0.6, 0.4, 0.4])
err1sup = array([0.4, 0.2, 0.3, 0.4, 0.5])            # Asymmetric error bars
p.plot(x0,y1,'rx')
p.plot(t,y1,'r')
p.errorbar(t,y1,[err1inf,err1sup],fmt = 'o')
p.grid(True)
p.show()