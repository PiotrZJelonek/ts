# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:57:44 2019

@author: Piotr Z. Jelonek
"""

from GlobalConstantsTS import gc_ts
import numpy as np
import time
from math import pi, cos, log, tan, floor, ceil, sqrt
import matplotlib.pyplot as plt
from scipy import integrate
from numpy.random import uniform

def params(alpha,beta,delta,mu,theta):
    
    # This procedure verifies if the parameters of a tempered stable distribution
    # have correct domains.
    #
    # INPUT
    # alpha - constant, real
    # beta  - constant, real
    # delta - constant, real
    # mu    - constant, real
    # theta - constant, real
    # 
    # RETURN
    # delta   - constant, postive (if gc_ts.normalised is True, this is the value of delta
    #           which yields a TS distribution with unit variance)
    # noerror - Boolean, equal to True if all the parameters lie in the correct 
    #           range, equal to zero otherwise
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    if (0 < alpha and alpha<2) and (-1 <= beta and beta <= 1) and (delta >0) and (theta >0):
        noerror=True
        
        # normalised is True iff std dev is equal to one
        if gc_ts.normalised:
            
            if alpha==1:
                vr=2*delta/(pi*theta)
            else:
                vr=alpha*(1-alpha)*((cos(0.5*pi*alpha))**(-1))*(delta**(alpha))*theta**(alpha-2)
                
            if gc_ts.outpt:
                print('\n------------------------- TS PARAMETERS --------------------------\n')
                print('With delta={0:6.4f} initial variance amounts to VarX={1:6.4f}.'.format(delta,vr))
            
            if vr!=1:
                if alpha==1:
                    delta=pi*theta/2
                else:
                    delta=(((theta**(2-alpha))*cos(0.5*pi*alpha))/(alpha*(1-alpha)))**(1/alpha)
                if gc_ts.outpt:
                    print('As the distribution is normalised, parameter delta is reset \nto delta={0:6.4f}.\n'.format(delta))
        
        # current values of parameters
        if gc_ts.outpt:
            print('Current parameter values: ')
            print('    alpha = {0:4.3f}'.format(alpha))
            print('    beta  = {0:4.3f}'.format(beta))
            print('    delta = {0:5.3f}'.format(delta))
            print('    mu    = {0:5.3f}'.format(mu))
            print('    theta = {0:5.3f}\n'.format(theta))

    else:
         print('Function ts.params::wrong range of parameters!\n')
         noerror=False

    return delta, noerror  

def rand_stab(alpha,beta,n=1):
    
    # This procedure returns pseudo-random draws from a stable random variable with
    # parameters alpha, beta, delta=1, and mu=0.
    #
    # INPUT
    # alpha - constant, in (0,2) interval
    # beta  - constant, in [-1,1] interval
    # n     - integer, a required number of draws, default value is 1
    # 
    # RETURN
    # x     - numpy.ndarray of the generated draws, its shape: (n,1)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    # rd=np.random.uniform(0,1,[n,2])
    w=-np.log(uniform(0,1,(n,1))) # log to base 'e'
    phi=(uniform(0,1,(n,1))-0.5)*pi
    cosphi = np.cos(phi)
    
    zeta = beta * tan(0.5*pi*alpha)
    if abs(alpha-1) > gc_ts.small:
        aphi = alpha * phi
        a1phi = (1 - alpha) * phi
        x=((np.sin(aphi) + zeta * np.cos(aphi))/cosphi)*(((np.cos(a1phi) + zeta * np.sin(a1phi))/(w*cosphi))**((1-alpha)/alpha))
    else:
        bphi = 0.5*pi + beta * phi
        x = (2/pi) *(bphi*np.tan(phi)- beta*np.log(0.5*pi*w*cosphi/bphi))
        if alpha !=1:
            x=x+zeta    
            
    return x

def getc(alpha):
    
    # This procedure returns a 0.01% quantile of a stable distribution with
    # parameters alpha, beta=1
    #
    # INPUT
    # alpha - constant, in (0,2) interval
    # 
    # RETURN
    # c     - constant, real
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    a=0.0001 # simulated 0.01% quantile
    x=sorted(rand_stab(alpha,1,gc_ts.large))
    c=-x[max(floor(a*gc_ts.large),1)]
    
    return c

def rand_ts_bm(alpha,theta,c,n=1):
    
    # This procedure returns pseudo-random draws from a tempered stable random variable 
    # with parameters alpha, beta=1, delta=1, mu=0, and theta.
    #
    # INPUT
    # alpha  - constant, in (0,2) interval
    # theta  - constant, positive
    # c      - constant, a cut-off value from the underlying stable distribution
    #          (here: 0.01% quantile)
    # n      - integer, a required number of draws, default value is 1
    # 
    # RETURN
    # v     - numpy.ndarray of the generated draws, its shape: (n,1)
    #
    # NOTE
    # The resulting random numbers are centred
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    draws=0; it=0; left=n
    v=np.full((n,1),np.nan)
    
    while left >0 and it<gc_ts.maxiter: 
        u=uniform(0,1,(left,1))
        x=rand_stab(alpha,1,left)
        ind=(u < np.exp(-theta*(x+c)))
        newdraws=ind.sum()
        v[draws:draws+newdraws,:]=x[ind].reshape((newdraws,1))
        draws=draws+newdraws
        left=left-newdraws
        it=it+1
        
    if it<gc_ts.maxiter:
        if alpha != 1:
            v=v-alpha*(theta**(alpha-1))/cos(0.5*pi*alpha)
        else:
            v=v+2*(log(theta)+1)/pi
    else:
        print('Function ts.rand_ts_bm::maximal admissible number of rejections reached!\n')

    return v

def rand_ts_mixture(alpha,beta,delta,mu,theta,n=1):
    
    # This procedure returns pseudo-random draws from a tempered stable random variable 
    # via a mixture algorithm.
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    # n      - integer, a required number of draws, default value is 1
    # 
    # RETURN
    # v      - numpy.ndarray of the generated draws (with n elements)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    if gc_ts.outpt:
        start_time=time.time()
    
    vp=delta*(0.5*(1+beta))**(1/alpha)
    vm=delta*(0.5*(1-beta))**(1/alpha)
    tp=theta*vp
    tm=theta*vm
    
    # finding c
    if alpha<1:
        c=0
    else:
        c=getc(alpha)

    # generating centred draws
    x1=0; x2=0

    if beta!=-1:
        x1=rand_ts_bm(alpha,tp,c,n)
    if beta!=1:
        x2=rand_ts_bm(alpha,tm,c,n)

    v=vp*x1-vm*x2+mu
    
    if gc_ts.outpt:
        print('-------------------- RANDOMISATION (MIXTURE) ---------------------\n')
        print('Mixture representation: Random numbers have been generated.')
        print('Mean of the generated draws: {0:.6f}.'.format(v.mean()))
        ms=round(1000*(time.time() - start_time),6)
        print('Total time: {0:.6f} miliseconds.\n'.format(ms))
    
    return v

def phi0(u):
    
    # This procedure returns the values of a characteristic function of a tempered stable
    # random variable, evaluated on vector u.
    #
    # INPUT
    # u      - real numpy.ndarray
    # 
    # RETURN
    # v      - complex numpy.ndarray (with a size matching the input)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    global alphat, betat, deltat, mut, thetat
    
    if alphat!=1:
        C=-0.5*(deltat**alphat)/cos(0.5*pi*alphat)
        phi=C*((1+betat)*(thetat-1j*u)**alphat + (1-betat)*(thetat+1j*u)**alphat - 2*thetat**alphat)
        mux=alphat*betat*(deltat**alphat)*(thetat**(alphat-1))/cos(0.5*pi*alphat)
    else:
        C=(1/pi)*deltat**alphat
        phi=C*((1+betat)*(thetat-1j*u)*np.log(thetat-1j*u) + (1-betat)*(thetat+1j*u)*np.log(thetat+1j*u) - 2*thetat*np.log(thetat))
        mux=-2*betat*deltat*(log(thetat)+1)/pi

    phi=np.exp(phi+1j*(mut-mux)*u)
    
    return phi

def pdf(alpha,beta,delta,mu,theta):
    
    # This procedure identifies the domain in which the pdf of a tempered stable random variable
    # exceeds the required value (gc_ts.small). Next it evaluates its pdf (via inverse Fourier 
    # transform) at (gc_ts.N) points evenly spaced across this domain.
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    # 
    # RETURN
    # x      - numpy.ndarray of points evenly spaced across the domain (with gc_ts.N elements)
    # y      - numpy.ndarray of the corresponding pdf values 
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    global alphat, betat, deltat, mut, thetat
    alphat=alpha; betat=beta; deltat=delta; mut=mu; thetat=theta
    en=np.arange(gc_ts.N)
    
    # finding [a,b] via Slow Fourier Transformation, as in Mittnik, Doganoglu & Chengyao (1999),
    # but their formulas are used directly
  
    x=1; b=1; k=gc_ts.N-1
    while abs(x) > gc_ts.small: 
        
        c=(1/(2*b))*(-1)**(k-0.5*gc_ts.N)
        u=(pi/b)*(en-gc_ts.N/2)
        y=phi0(u)
        
        # if u contains zero, y contains NaN
        y[np.isnan(y)]=1
        
        d=1
        for j in range(2,gc_ts.N):
            d=-d
            y[j]=d*y[j]
        
        y=c*y*np.exp(-2*pi*1j*k*en/gc_ts.N)
        x=np.sum(np.array(sorted(y.real,key=abs)))
        b=b*1.25
        
    x=1; a=-1; k=0
    while abs(x) > gc_ts.small: # should there be abs here?
        
        c=(1/(2*a))*(-1)**(k-0.5*gc_ts.N)
        u=(pi/a)*(en-gc_ts.N/2)
        y=phi0(u)
        
        # if u contains zero, y contains NaN
        y[np.isnan(y)]=1
        
        d=1
        for j in range(2,gc_ts.N):
            d=-d
            y[j]=d*y[j]
            
        y=c*y*np.exp(-2*pi*1j*k*en/gc_ts.N)
        x=np.sum(np.array(sorted(y.real,key=abs)))
        a=a*1.25    
            
    b=ceil(max([-a,b]))    
    a=-b
    
    # Modification of Mittnik, Doganoglu & Chengyao (1999) that allows for
    # asymmetric interval (i.e. [a,b] instead of [-b,b]) - the first run
    u=(2*pi/(b-a))*(en-gc_ts.N/2)
    x=phi0(u)

    # if u contains zero, x contains NaN
    x[np.isnan(x)]=1
    
    # computing y's
    d=1
    c=(-1)**(-2*a/(b-a))
    for j in range(2,gc_ts.N):
        d=d*c
        x[j]=d*x[j] 
    y=np.fft.fft(x); y=y*(-1)**(a*gc_ts.N/(b-a))
    d=1
    for j in range(2,gc_ts.N):
        d=-d
        y[j]=d*y[j]
    y=y/(b-a)
    
    # correcting numerical errors
    y=y.real
    
    # computing x's
    x=a+(b-a)*np.arange(gc_ts.N)/gc_ts.N
      
    # Now we find a better interval for X
    idy=np.column_stack((np.arange(gc_ts.N).astype(int),y)) # concatenating 2 columns    
    idy=idy[np.argsort(-idy[:,1])]; id=idy[0,0].astype(int) # sorting according to 2nd column, descending order (-)
    j=id
    while y[j]>gc_ts.small:
        j-=1
    a=floor(x[j])
    j=id
    while y[j]>gc_ts.small:
        j+=1
    b=ceil(x[j])    
    
    # Modification of Mittnik, Doganoglu & Chengyao (1999) - the final run
    u=(2*pi/(b-a))*(en-gc_ts.N/2)
    x=phi0(u)

    # if u contains zero, x contains NaN
    x[np.isnan(x)]=1
    
    # computing y's
    d=1
    c=(-1)**(-2*a/(b-a))
    for j in range(2,gc_ts.N):
        d=d*c
        x[j]=d*x[j] 
    y=np.fft.fft(x); y=y*(-1)**(a*gc_ts.N/(b-a))
    d=1
    for j in range(2,gc_ts.N):
        d=-d
        y[j]=d*y[j]
    y=y/(b-a)
    
    # correcting numerical errors
    y=y.real
    
    # computing x's
    x=a+(b-a)*np.arange(gc_ts.N)/gc_ts.N
    
    if gc_ts.outpt:
        # plotting the output
        fig=plt.figure(); ax=fig.add_subplot(1,1,1)
        ax.plot(x,y,'b')
        ax.set_title('PDF OF TS RANDOM VARIATE')  
        ax.set_ylabel('PDF VALUE')
        ax.set_xlabel('ORDINATE')
        ax.set_xlim([min(x),max(x)])
        ax.set_ylim([0,1.1*max(y)])
        
        # support       
        print('---------------------------- TS PDF ------------------------------\n')
        print('Pdf of TS is trimmed to [{0:2.0f},{1:2.0f}].'.format(a,b))
                
        # computing mean X
        meanx=(x[1]-x[0])*np.sum(x*y)
        print('Approximate mean of TS distributed random variable: {0:5.4f}.\n'.format(meanx))

    return x, y 

def cdf(alpha,beta,delta,mu,theta):
    
    # This procedure calculates cdf of a tempered stable random variable using its 
    # (Fourier transformed) pdf. Next it corrects the numerical errors.
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    # 
    # RETURN
    # x      - numpy.ndarray of points evenly spaced across the domain (with gc_ts.N elements)
    # z      - numpy.ndarray of the corresponding cdf values  
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    # supressing output for the pdf
    outpt=gc_ts.outpt; gc_ts.outpt=False
    x,y=pdf(alpha,beta,delta,mu,theta)
    gc_ts.outpt=outpt
    
    z=(x[1]-x[0])*y.cumsum()
    z=z-z[0]
    z=z/z[gc_ts.N-1]
    
    if gc_ts.outpt:
        # plotting the output
        fig=plt.figure(); ax=fig.add_subplot(1,1,1)
        ax.plot(x,z,'b')
        ax.set_title('CDF OF TS RANDOM VARIATE')  
        ax.set_ylabel('CDF VALUE')
        ax.set_xlabel('ORDINATE')
        ax.set_xlim([min(x),max(x)])
        ax.set_ylim([0,1.1*max(z)])
        plt.show() # <- print now, not after the code is executed
    
    return x, z


def rand_ts_inv(alpha,beta,delta,mu,theta,n=1):
    
    # This procedure returns pseudo-random draws from a tempered stable random variable 
    # via a cdf inversion algorithm (requires inverse Fourier transform 
    # to obtain the pdf).
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    # n      - integer, a required number of draws, default value is 1
    # 
    # RETURN
    # v      - numpy.ndarray of the generated draws (with n elements)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    if gc_ts.outpt:
        start_time=time.time()
    
    # supressing output for the cdf
    outpt=gc_ts.outpt; gc_ts.outpt=False
    x,z=cdf(alpha,beta,delta,mu,theta)
    gc_ts.outpt=outpt
    
    # x's contain only left ends of intervals, but we also need 
    # the right end of the final interval
    h=x[1]-x[0]
    x=np.append(x,[x[-1]+h],axis = 0)
    z=np.append([0.0],z,axis=0)
    dz=z[1:]-z[0:-1]
    
    v=np.zeros(n)
    id=np.arange(gc_ts.N+1)
    U=sorted(np.random.uniform(0,1,n))
    
    down=0
    for j in range(0,n):
        u=U[j]; up=down+1
        # run only if previous interval is not sufficient
        if u > z[up]:
            up=id[-1]; cur=np.round(0.5*(down+up)).astype(int)
            while cur!=up and cur!=down:
                if u<z[cur]:
                    up=cur
                    cur=ceil(0.5*(down+up))
                else:
                    down=cur
                    cur=floor(0.5*(down+up))

        if u==z[up]:
            v[j]=x[up]
        elif u==z[down]:
            v[j]=x[down]
        else:
            v[j]=x[down]+h*(u-z[down])/dz[down]
    
    # mixing draws
    v=v[np.argsort(np.random.uniform(0,1,n))]        

    if gc_ts.outpt:
        print('----------------- RANDOMISATION (CDF INVERSION) ------------------\n')
        print('Cdf Inversion: Random numbers have been generated.')
        print('Mean of the generated draws: {0:6.5f}.'.format(v.mean()))
        ms=round(1000*(time.time() - start_time),6)
        print('Total time: {0:.6f} miliseconds.\n'.format(ms))
    
    return v.reshape((n,1))

def mod_phi2(u):
    
    # This procedure returns modulus of a second order derivative of a
    # characteristic function of a centred (mu=0) tempered stable 
    # random variable.
    #
    # INPUT
    # u        - real numpy.ndarray 
    # 
    # RETURN
    # abs(phi) - numpy.ndarray of absolute values of the cf (with a size matching the input)
    
    global alphat, betat, deltat, mut, thetat
    if alphat!=1:
        phi=-((0.5*alphat*deltat**alphat)/cos(0.5*pi*alphat))*(((0.5*alphat*deltat**alphat)/cos(0.5*pi*alphat))*(((1+betat)*(thetat-1j*u)**(alphat-1) - \
        (1-betat)*(thetat+1j*u)**(alphat-1)-2*betat*(thetat**(alphat-1)))*(2)) +(1-alphat)*((1+betat)*(thetat-1j*u)**(alphat-2) + \
        (1-betat)*(thetat+1j*u)**(alphat-2)))
    else:
        phi=-((deltat**alphat)/pi)*(((deltat**alphat)/pi)*((1+betat)*log(thetat-1j*u) - (1-betat)*log(thetat+1j*u) - \
        2*betat*log(thetat))**2+ 2*(thetat+1j*betat*u)/(thetat**2+u**2))
    phi=phi*abs(phi0(u))
    
    return abs(phi)

def getcform(x,y):
    
    # This procedure outputs a cubic form interpolation of a function, values of which
    # (y) are known only at given nodes (x)
    #
    # INPUT
    # x                     - real numpy.ndarray (of size gc_ts.N), arguments
    # y                     - real numpy.ndarray (of size gc_ts.N), values of the function
    # 
    # RETURN
    # np.array([a,b,c,d]).T - real (gc_ts.N -1 x 4) numpy.ndarray, parameters of the cubic form
    #
    # REFERECE: Richard L. Burden, J. Douglas  Faires, "Numerical Analysis", 6th edition, p. 148

    dx=x[1:]-x[0:-1]; n=x.size-1; h=dx; a=y
    
    al=3*((a[2:]-a[1:-1])/h[1:]-(a[1:-1]-a[0:-2])/h[0:-1])
    l=np.zeros(n+1); l[0]=1; l[n]=1; m=np.zeros(n); z=np.zeros(n+1)
    
    for i in range(1,n):
        l[i]=2*(x[i+1]-x[i-1])-h[i-1]*m[i-1]
        m[i]=h[i]/l[i]  
        z[i]=(al[i-1]-h[i-1]*z[i-1])/l[i]
        
    c=np.zeros(n+1); b=np.zeros(n); d=np.zeros(n); i=n-2
    while i > 0:
        c[i]=z[i]-m[i]*c[i+1]
        b[i]=(a[i+1]-a[i])/h[i]-h[i]*(c[i+1]+2*c[i])/3
        d[i]=(c[i+1]-c[i])/(3*h[i])
        i=i-1
        
    a=a[:-1]; c=c[:-1]     
    
    return np.array([a,b,c,d]).T

def rand_ts_devroye(alpha,beta,delta,mu,theta,n=1):
    
    # This procedure returns pseudo-random draws from a tempered stable random variable 
    # via an algorithm by Devroye (1981).
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    # n      - integer, a required number of draws, default value is 1
    # 
    # RETURN
    # v      - numpy.ndarray of generated draws (with n elements)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    global alphat, betat, deltat, mut, thetat
    alphat=alpha; betat=beta; deltat=delta; mut=mu; thetat=theta
    mod_phi0 = lambda u: abs(phi0(u))
    
    if gc_ts.outpt:
        start_time=time.time()
    
    # find centred draws (it is easier and possibly speeds up)
    mut=0
    
    # find support of phi0 
    xb=-1; xt=1
    while mod_phi0(xb) > gc_ts.small:
        xb=xb*1.05
    while mod_phi0(xt) > gc_ts.small:
        xt=xt*1.05
 
    ## fixed-order Gauss-Lobatto quadrature
    #Q=integrate.fixed_quad(mod_phi0,xb,xt,n=19)
     
    # adaptive quadrature with a fixed tolerance
    Q=integrate.quadrature(mod_phi0,xb,xt,args=(),tol=gc_ts.small,rtol=gc_ts.small,maxiter=gc_ts.maxorder)
    
    # find support of phi2
    xb=-1; xt=1
    while mod_phi2(xb) > gc_ts.small:
        xb=xb*1.05
    while mod_phi2(xt) > gc_ts.small:
        xt=xt*1.05
    
    R=integrate.quadrature(mod_phi2,xb,xt,args=(),tol=gc_ts.small,rtol=gc_ts.small,maxiter=gc_ts.maxorder)
    
    C1=Q[0]/(2*pi) # in Devroye (1981) this is c
    C2=R[0]/(2*pi) # in Devroye (1981) this is k 
    C3=4*sqrt(C1*C2) # in Devroye (1981) this is A
    q=sqrt(C2/C1)
    
    # a discretised pdf
    outpt=gc_ts.outpt; gc_ts.outpt=False
    x,y=pdf(alpha,beta,delta,mu,theta)
    gc_ts.outpt=outpt
    
    minx=min(x); maxx=max(x); h=x[1]-x[0]; ih=h**(-1)
    
    # obtaining a cubic form for evenly spaced x's
    zz = getcform(x,y)
    
    # here go the generated numbers
    v=np.zeros(n)
    
    # generating draws from centred TS
    for i in range(0,n):
        
        flag=False
        
        while not(flag):
            
            U=np.random.uniform(0,1,3)
            V=2*U[:-1]-1; x=q*V[0]/V[1]
            
            if x > minx and x < maxx:
                
                # evaluating a cubic form at x
                bin=floor(ih*(x-minx))
                cx=x-(minx+(bin-1)*h)
                fx=zz[bin,0]+cx*(zz[bin,1]+cx*(zz[bin,2]+cx*zz[bin,3]))
                
                if abs(V[0]) < abs(V[1]):
                    if C1*U[2] < fx:
                        flag=True
                else:
                    if C2*U[2] < fx*x**2:
                        flag=True
            v[i]=x                
                
    # reversing centring
    mut=mu
    v=v+mut
    
    if gc_ts.outpt:
        print('-------------------- RAMDOMISATION (DEVROYE) ---------------------\n')
        print('Devroye (1981): Random numbers have been generated.')
        print('Mean of the generated draws: {0:6.5f}.'.format(v.mean()))
        print('Average number of calls for each generated number: {0:5.3f}.'.format(C3))
        ms=round(1000*(time.time() - start_time),6)
        print('Total time: {0:.6f} miliseconds.\n'.format(ms))
    
    return v.reshape((n,1))

def moments_empiric(v):
    
    # This procedure returns sample moments of independent tempered stable draws.
    #
    # INPUT
    # v - numpy.ndarray (with n elements) of i.i.d draws from a tempered stable distribution 
    # 
    # RETURN
    # m - numpy.ndarray of first 6 sample moments about the origin
    # k - numpy.ndarray, of first 6 sample cumulants
    # c - numpy.ndarray with, respectively, sample: mean, variance, skewness and excess kurtosis
    #
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    n=v.shape[0] 
    m=np.zeros(6)
    m[0]=(1/n)*v.sum()
    m[1]=(1/n)*((v**2).sum())
    m[2]=(1/n)*((v**3).sum())
    m[3]=(1/n)*((v**4).sum())
    m[4]=(1/n)*((v**5).sum())
    m[5]=(1/n)*((v**6).sum())
    
    k=np.zeros(6)
    k[0]=m[0]
    k[1]=m[1]-k[0]**2
    k[2]=m[2]-3*k[1]*k[0]-k[0]**3
    k[3]=m[3]-4*k[2]*k[0]-3*k[1]**2-6*k[1]*k[0]**2-k[0]**4 
    k[4]=m[4]-5*k[3]*k[0]-10*k[2]*k[1]-10*k[2]*k[0]**2-15*k[0]*k[1]**2-10*k[1]*k[0]**3-k[0]**5
    k[5]=m[5]-6*k[4]*k[0]-15*k[3]*k[1]-15*k[3]*k[0]**2-10*k[2]**2-60*k[2]*k[1]*k[0]-20*k[2]*k[0]**3-15*k[1]**3-45*(k[1]*k[0])**2-15*k[1]*k[0]**4-k[0]**6

    c=np.zeros(4)
    c[0:2]=k[0:2]
    c[2]=k[2]/(k[1]**1.5)
    c[3]=k[3]/(k[1]**2.0)
    
    if gc_ts.outpt:
        print('------------------------ EMPIRIC RESULTS -------------------------\n')
        print('First six cumulants:')
        for i in range(0,6): print('k{0:1.0f}: {1:6.5f}'.format(i+1,k[i]))
        print('\nFirst six moments about the origin:')
        for i in range(0,6): print('m{0:1.0f}: {1:6.5f}'.format(i+1,m[i]))
        print('\nExpected value, variance, skewness, excess kurtosis:')
        print('mean            = {0:6.5f}'.format(c[0]))
        print('variance        = {0:6.5f}'.format(c[1]))
        print('skewness        = {0:6.5f}'.format(c[2]))
        print('excess kurtosis = {0:6.5f}\n'.format(c[3]))    
    
    return m, k, c
    
def estimate(v):
    
    # This procedure estimates parameters of a tempered stable distribution
    # from given sample.
    #
    # INPUT
    # v     - numpy.ndarray (with n elements) of i.i.d draws from a tempered stable distribution 
    #
    # RETURN
    # ahat  - constant, estimate of alpha
    # bhat  - constant, estimate of beta
    # dhat  - constant, estimate of delta
    # mhat  - constant, estimate of mu
    # that  - constant, estimate of theta
    #
    # WARNING
    # This procedure relies on sample moments of higher order. It requires a really large sample.
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
      
    # to do: alpha=1, parameters outside of boundary, multivariate estimation
    
    ## scaling
    #scaling=abs(v).max()
    #v=v/scaling
    
    # standardising the sample
    meanv=v.mean()
    varv=v.var()
    stdv=sqrt(varv)
    v=(v-meanv)/stdv
    
    # cumulants of the standardised sample
    outpt=gc_ts.outpt; gc_ts.outpt=False
    m, k, c = moments_empiric(v)
    gc_ts.outpt=outpt    
    
    # parameters of the standardised sample
    ahat=2*(1+1/(1-k[4]/(k[3]*k[2])))
    that=sqrt((2-ahat)*(3-ahat)/k[3])
    bhat=that*k[2]/(2-ahat)
    
    # reversing standardisation
    mhat=meanv
    dhat=(varv*cos(0.5*pi*ahat)/(ahat*(1-ahat)*that**(ahat-2)))**(1/ahat)
    
    ## reverting scaling
    #dhat=dhat*scaling
    #mhat=mhat*scaling
    #that=that/scaling
    
    if gc_ts.outpt:
        print('---------------------- ESTIMATION RESULTS ------------------------\n')
        print('Estimated parameters:')
        print('    alpha = {0:6.5f}'.format(ahat))
        print('    beta  = {0:6.5f}'.format(bhat))
        print('    delta = {0:6.5f}'.format(dhat))
        print('    mu    = {0:6.5f}'.format(mhat))
        print('    theta = {0:6.5f}\n'.format(that))
    
    return ahat,bhat,dhat,mhat,that

def moments_theoretic(alpha,beta,delta,mu,theta):
    
    # This procedure returns theoretic moments of independent tempered stable draws.
    #
    # INPUT
    # alpha  - constant, in (0,2)
    # beta   - constant, in [-1.1]
    # delta  - constant, positive
    # mu     - constant, real
    # theta  - constant, positive
    #
    # RETURN
    # m - numpy.ndarray of first 6 theoreti moments about the origin
    # k - numpy.ndarray, of first 6 theoretic cumulants
    # c - numpy.ndarray with, respectively, theoretic: mean, variance, skewness and excess kurtosis
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    # cumulants, moments, classical characteristics
    k=np.zeros(6); m=np.zeros(6); c=np.zeros(4)
    
    # first six cumulants
    if alpha !=1:
        k[5]=alpha*(1-alpha)*(2-alpha)*(3-alpha)*(4-alpha)*(5-alpha)*(delta**alpha)*(theta**(alpha-6))/cos(0.5*pi*alpha)
        k[4]=alpha*(1-alpha)*(2-alpha)*(3-alpha)*(4-alpha)*beta*(delta**alpha)*(theta**(alpha-5))/cos(0.5*pi*alpha)
        k[3]=alpha*(1-alpha)*(2-alpha)*(3-alpha)*(delta**alpha)*(theta**(alpha-4))/cos(0.5*pi*alpha)
        k[2]=alpha*(1-alpha)*(2-alpha)*beta*(delta**alpha)*(theta**(alpha-3))/cos(0.5*pi*alpha)
        k[1]=alpha*(1-alpha)*(delta**alpha)*(theta**(alpha-2))/cos(0.5*pi*alpha)
    else:
        k[5]=48*(delta/pi)*theta**(-5)
        k[4]=12*(beta*delta/pi)*theta**(-4)
        k[3]=4*(delta/pi)*theta**(-3)
        k[2]=2*(beta*delta/pi)*theta**(-2)
        k[1]=2*(delta/pi)*theta**(-1)
    k[0]=mu
    
    # first six moments about the origin
    m[0]=k[0]
    m[1]=k[1]+k[0]**2
    m[2]=k[2]+3*k[1]*k[0]+k[0]**3
    m[3]=k[3]+4*k[2]*k[0]+3*k[1]**2+6*k[1]*k[0]**2+k[0]**4
    m[4]=k[4]+5*k[3]*k[0]+10*k[2]*k[1]+10*k[2]*k[0]**2+15*k[0]*k[1]**2+10*k[1]*k[0]**3+k[0]**5
    m[5]=k[5]+6*k[4]*k[0]+15*k[3]*k[1]+15*k[3]*k[0]**2+10*k[2]**2+60*k[2]*k[1]*k[0]+20*k[2]*k[0]**3+15*k[1]**3+ \
    45*(k[1]*k[0])**2+15*k[1]*k[0]**4+k[0]**6
    
    # % expectation, variance, skewness, excess kurtosis
    c[0]=k[0]; c[1]=k[1]
    if alpha !=1:
        c[2]=(2-alpha)*beta*sqrt(cos(0.5*pi*alpha)/(alpha*(1-alpha)*(delta*theta)**alpha))
        c[3]=(2-alpha)*(3-alpha)*cos(0.5*pi*alpha)/(alpha*(1-alpha)*(delta*theta)**alpha)
    else:
        c[2]=sqrt(pi/2)*beta/sqrt(theta*delta)
        c[3]=pi/(theta*delta)
         
    if gc_ts.outpt:
        print('----------------------- THEORETIC RESULTS ------------------------\n')
        print('First six cumulants:')
        for i in range(0,6): print('k{0:1.0f}: {1:6.5f}'.format(i+1,k[i]))
        print('\nFirst six moments about the origin:')
        for i in range(0,6): print('m{0:1.0f}: {1:6.5f}'.format(i+1,m[i]))
        print('\nExpected value, variance, skewness, excess kurtosis:')
        print('mean            = {0:6.5f}'.format(c[0]))
        print('variance        = {0:6.5f}'.format(c[1]))
        print('skewness        = {0:6.5f}'.format(c[2]))
        print('excess kurtosis = {0:6.5f}\n'.format(c[3]))

    return m, k, c

def gsf(mu,sigma,x):
    
    # This procedure returns the pdf of a normal distribution.
    #
    # INPUT
    # mu     - mean
    # sigma  - standard deviation
    # x      - real numpy.ndarray of values (n entries)
    #
    # RETURN
    # y      - numpy.ndarray of pdf values (with a size matching the input)
    #
    # Written by: Piotr Z. Jelonek in March 2019, contact: piotr.z.jelonek@gmail.com
    
    y=(1/(sigma*sqrt(2*pi)))*np.exp(-(0.5/sigma**2)*(x-mu)**2)
    
    return y