# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:39:44 2018

@author: Chad
"""

#Code for CP7.3.2a
# CP7.3.2a
# Use collocation to solve the BVP
import numpy as np
import matplotlib . pylab as plt
# define the basis functions and the second derivatives
def phi (t , n ):

    w = np . zeros (( t . size , n ))
    for k in np . arange ( n ):
        w [: , k ] = t ** k
        #print(w [: , k ])
    return w

def ddphi (t , n ):
    w = np . zeros (( t . size , n ))
    w [: ,0] = 0
    w [: ,1] = 0
    w [: ,2] = 2
    for k in np . arange (3 , n ):
        #makes an array from 3 to n with the pre computed derivative
        w [: , k ] = k *( k -1)* t **( k -2)
    return w

 # implement the collocation method
def bvpcol ( inter , bv , n ):
    [a , b ] = inter
    t = np . linspace (a , b , n , endpoint = True )
    [ ya , yb ] = bv
    # build system matrix
    m = ddphi (t , n ) + (( np . pi **2)/9)* phi (t , n )
    m [0 , :] = phi ( t [0] , n )[0]
    m [ -1 ,:] = phi ( t [n -1] , n )[0]
    # build rhs vector
    d = np . zeros ( n )
    d [ 0] = ya
    d [ -1] = yb
    # solve system
    w = np . linalg . solve (m , d )
    return w

# exact solution
def exact ( t ):
    return 3* np . sin ( np . pi * t /3) - np . cos ( np . pi * t /3)
# initialize the BVP data
a = 0
b = 1.5
bv = [ -1 , 3]

# c8 is the list of coefficients in yhat(t) = c1 phi1(t) + ... + c8 phi8(t)
n = 8
c8 = bvpcol ([ a , b ] , bv , n )

# c16 is the list of coefficients in yhat(t) = c1 phi1(t) + ... + c16 phi16(t)
n = 16
c16 = bvpcol ([ a , b ] , bv , n )

# plot solutions
fig = plt . figure (1 , figsize = (8 , 6))
plt . xlim ([ -0.1 , 1.6])
plt . ylim ([ -1.1 , 3.1])
t = np . linspace (a , b , 20 , endpoint = True )
plt . plot (t , exact ( t ) , 'b-', linewidth =2.0 , label ='Exact ')
# polyval evaluates yhat(t) at every point in t
# need to swap order of coefficients for polyval, which is what c8[::−1] does
# see documentation for numpy.polyval
w8 = np . polyval ( c8 [:: -1] , t )
plt . plot (t , w8 , 'ro ', linewidth =2.0 , label ='Approx , $n =8$')
w16 = np . polyval ( c16 [:: -1] , t )
plt . plot (t , w16 , 'ro', linewidth =2.0 , label ='Approx , $n =16 $')
plt . legend ( loc='upper right', fontsize =14)
plt . title ('Solutions ', fontsize =20 , color ='#22222f')
plt . xlabel ('$t$ ', fontsize =20)
plt . ylabel ('$y(t) , y_8 (t) , y_ {16}( t)$', fontsize =18)
plt . show ()
fig . savefig ('testplot1.png')

# plot errors
fig = plt . figure (1 , figsize = (8 , 6))
plt . xlim ([ -0.1 , 1.6])
plt . ylim ([10**( -17.0) , 10**( -5.0)])
t = np . linspace ( a +0.001 , b -0.001 , 200 , endpoint = True )
et = exact ( t )
w8 = np . polyval ( c8 [:: -1] , t )
plt . semilogy (t , np .abs ( et - w8 ) , 'r-', linewidth =2.0 , label ='$n = 8$')
w16 = np . polyval ( c16 [:: -1] , t )
plt . semilogy (t , np .abs ( et - w16 ) , 'b-', linewidth =2.0 , label ='$n = 16$')
plt . legend ( loc='upper right', fontsize =16)
plt . title ('Error ', fontsize =20 , color ='#22222f')
plt . xlabel ('$t$ ', fontsize =20)
plt . ylabel ('Log ( error )', fontsize =18)
plt . show ()
fig . savefig ('testplot.png')