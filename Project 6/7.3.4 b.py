# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:17:54 2018

@author: Chad
"""

#CP7.3.4b
# solve the BVP
# y'' + 2y' − 3y = 0, y(0) = e^3, y(1) = 1
# using the finite element method
# y(t) = e^{3−3t}

import numpy as np
import scipy . sparse as sp
import scipy . sparse . linalg as spl
import matplotlib . pylab as plt

def bvpfem ( inter , bv , n ):
    a = inter [0]
    b = inter [1]
    ya = bv [0]
    yb = bv [1]
    h = (b - a )/ float ( n +1)
# build system matrix
    alpha = 1.0/ h - h /2.0
    beta = -2.0/ h - 2.0* h
    e = np . ones ( n )
    M = sp . dia_matrix ( ([( alpha -1)* e , beta *e , ( alpha +1)* e ] ,[ -1 ,0 ,1]) , \
                         shape =( n , n ) ). tocsr ()
# print(M.todense())

# build rhs vector
    d = np . zeros ( n )
    d [ 0] = - ya *( alpha -1)
    d [ -1] = - yb *( alpha +1)
# print(d)

    w = spl . spsolve (M , d )
    return w

def exact ( t ):
    return np . exp (3 - 3* t )

a = 0
b = 1
bv = [ np . exp (3) , 1]

n = 8
w8 = bvpfem ([ a , b ] , bv , n )
w8 = np . hstack (( bv [0] , w8 , bv [1]))
t8 = np . linspace (a , b , n +2 , endpoint = True )

n = 16
w16 = bvpfem ([ a , b ] , bv , n )
w16 = np . hstack (( bv [0] , w16 , bv [1]))
t16 = np . linspace (a , b , n +2 , endpoint = True )

fig = plt . figure (1 , figsize = (8 , 6))
plt . xlim ([ -0.1 , 1.1])
plt . ylim ([ -0.1 , 22.0])
t = np . linspace (a , b , 200 , endpoint = True )
plt . plot (t , exact ( t ) , 'b-', linewidth =2.0 , label ='Exact ')
plt . plot ( t8 , w8 , 'ro ', linewidth =2.0 , label ='Approx , $n = 8$')
plt . plot ( t16 , w16 , 'r--', linewidth =2.0 , label ='Approx , $n = 16$')
plt . legend ( loc='upper right', fontsize =14)
plt . title ('Solutions ', fontsize =30 , color ='#22222f')
plt . xlabel ('$t$ ', fontsize =24)
plt . ylabel ('$y(t) , y_8 (t) , y_ {16}( t)$', fontsize =24)
plt . show ()
fig . savefig ('testplot1.png')

fig = plt . figure (1 , figsize = (8 , 6))
plt . xlim ([ -0.1 , 1.1])
t = np . linspace (a , b , 500 , endpoint = True )
et = exact ( t )

w8interp = np . interp (t , t8 , w8 )
plt . semilogy (t , np .abs ( et - w8interp ) , 'r-', linewidth =1.5 , \
label ='$n = 8$')

w16interp = np . interp (t , t16 , w16 )
plt . semilogy (t , np .abs ( et - w16interp ) , 'b-', linewidth =1.5 , \
label ='$n = 16$')

plt . legend ( loc='upper right', fontsize =14)
plt . title ('Error ', fontsize =30 , color ='#22222f')
plt . xlabel ('$t$ ', fontsize =24)
plt . ylabel ('Log ( error )', fontsize =24)
plt . show ()
fig . savefig ('testplot.png')

# remark: the "spikes" in the error curve are due to the error alternating
# between positive and negative values, and then applying the np.abs,
# which flips all the negative aprts to positive