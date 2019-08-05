# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:32:30 2019

@author: Piotr Z. Jelonek
"""

from GlobalConstantsTS import gc_ts
import ts 
import matplotlib.pyplot as plt
import seaborn as sns

# clear the terminal
print(chr(27) + "[2J")

# number of random draws
n=100000

# distribution parameters
alpha=1.3
beta=0.5
delta=1
mu=0
theta=0.5

# verifying & adjusting parameters   
delta, noerror=ts.params(alpha,beta,delta,mu,theta)

# randomisation, alpha stable distributions
w=ts.rand_stab(alpha,beta,n)

c=ts.getc(alpha)

v1=ts.rand_ts_mixture(alpha,beta,delta,mu,theta,n)

v2=ts.rand_ts_inv(alpha,beta,delta,mu,theta,n)

v3=ts.rand_ts_devroye(alpha,beta,delta,mu,theta,n)

# supressing the output 
outpt=gc_ts.outpt; gc_ts.outpt=False

# theoretic moments
m, k, c =ts.moments_theoretic(alpha,beta,delta,mu,theta)

# empiric moments
m1, k1, c1 =ts.moments_empiric(v1)
m2, k2, c2 =ts.moments_empiric(v2)
m3, k3, c3 =ts.moments_empiric(v3)

# theoretic vs. empirical moments in the three randomisation methods 
print('-------------------- MOMENTS ABOUT THE ORIGIN --------------------\n')
print('  Theoretic vs. Mixture vs. Inversion vs. Devroye')
for i in range(0,6): print('m{0:1.0f}: {1:7.5f}     {2:7.5f}     {3:7.5f}     {4:7.5f}'.format(i+1,m[i],m1[i],m2[i],m3[i]))

# theoretic vs. empirical moments in the three randomisation methods
print('\n--------------------------- CUMULANTS ----------------------------\n')
print('  Theoretic vs. Mixture vs. Inversion vs. Devroye')
for i in range(0,6): print('k{0:1.0f}: {1:7.5f}     {2:7.5f}     {3:7.5f}     {4:7.5f}'.format(i+1,k[i],k1[i],k2[i],k3[i]))

# theoretic vs. empirical characteristics in the three randomisation methods
print('\n------------------------ CHARACTARISTICS -------------------------\n')
print('                  Theoretic vs. Mixture vs. Inversion vs. Devroye')
print('mean            : {0:7.5f}       {1:7.5f}     {2:7.5f}       {3:7.5f}'.format(c[0],c1[0],c2[0],c3[0]))
print('variance        : {0:7.5f}       {1:7.5f}     {2:7.5f}       {3:7.5f}'.format(c[1],c1[1],c2[1],c3[1]))
print('skewness        : {0:7.5f}       {1:7.5f}     {2:7.5f}       {3:7.5f}'.format(c[2],c1[2],c2[2],c3[2]))
print('excess kurtosis : {0:7.5f}       {1:7.5f}     {2:7.5f}       {3:7.5f}\n'.format(c[3],c1[3],c2[3],c3[3]))

# pdf
x,y=ts.pdf(alpha,beta,delta,mu,theta)
gy=ts.gsf(c[0],c[1],x)

# restoriing the output 
gc_ts.outpt=outpt

# graphs
print('---------------------------- GRAPHS ------------------------------\n')
fig=plt.figure(); ax=fig.add_subplot(1,1,1)
ax.plot(x,gy,'k--',label='GAUSSIAN')
ax.plot(x,y,'r',label='TS')
ax.set_title('PDF OF TS VS. GAUSSIAN RANDOM VARIATE')  
ax.set_ylabel('PDF VALUE')
ax.set_xlabel('ORDINATE')
ax.set_xlim([min(x),max(x)])
ax.set_ylim([0,1.1*max(max(y),max(gy))])
ax.legend(loc='best') 
plt.show() # <- print now, not after the code is executed

# cdf
x,z=ts.cdf(alpha,beta,delta,mu,theta)

# ahat, bhat, dhat, mhat, that = ts.estimate(v1)