""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# FileCatchThrow.py:  throws and catches exception 

#program with mistake to see action of exception
import sys
import math
r = 2
circum = 2.* math.pi* r                                # Calculate circum
A = math.pi*r**2                                            # Calculate A
try: 
    q = open("ThrowCatch.dat",'w')                # Intentional error 'r'
                                                 # Replace r' to 'w', run
except  IOError:
    print 'Cannot open file'
else:     
     q.write("r = %9.6f, length = %9.6f, A= %9.6f "%(r, circum, A))
     q.close()
     print 'output in ThrowCatch.out'
     #catch(IOException ex){ex.printStackTrace(); }               # Catch
