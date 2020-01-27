"""
Project Title: Simulating a galaxy using Density Wave Theory.

orbit_plotting.py: Shows the orbits for a given number of stars over a given 
number of evolutions.

Author: Michael Wilson
Student No.: 16364513
Date: 02/12/2019
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#define style of background used for subsequent plots/animations
plt.style.use('dark_background')

#input initial parameters

#e1=Inner E
e1=0.8
#e2=Outer E
e2=1.0
#rcore=Inner core radius
rcore=6000
#rgalaxy=Galaxy radius
rgalaxy=15000
#rdist=Galaxy diameter
rdist=2*rgalaxy
#N=number of stars in galaxy
Nstars=int(input('Please input a value for the number of star orbits: '))
#velocity 1 is the velocity that the stars move at
velocity1=5e-6
#velocity 2 is the velocity the spiral arms move at (<velocity1)
velocity2=1e-7
#timesteps=number of timesteps to iterate over
timestep=1e6
#I0=intensity at centre of galaxy relative to other stars
I0=1

#seed a random number generator
np.random.seed(None)

def semi_major_axis(N):
    """
    Function semi_major_axis sets up the semi-major axis, a, for each star.
    These follow a random truncated normal distribution.
    """
    #set up an array to hold orbit values
    a_vals=np.empty(N)
    
    #use stats.truncnorm to give orbits a truncated normal distribution
    #set up values for mean and standard deviation (sigma)
    lower_bound=-rgalaxy
    upper_bound=rgalaxy
    mean=0
    sigma=rgalaxy/2
    
    #define an object T for the truncated normal distribution
    T=stats.truncnorm((lower_bound-mean)/sigma, (upper_bound-mean)/sigma,\
                    loc=mean, scale=sigma)
    
    #fill a_vals with random values coming from T
    a_vals=T.rvs(N)
    
    return a_vals
    
def object_angle(N):
    """
    Function star_angle calculates and returns the angle 'theta' for each
    star.
    """
    #assign an angle to each star
    #star_angles=array of angles of stars
    theta_angles=np.empty(N)
    
    for i in range(N):
        theta_angles[i]=np.random.uniform(0,360)
        #convert values in star_angles to radians
        theta_angles[i]=np.radians(theta_angles[i])
        
    return theta_angles

def ellipticity(N,a_vals,rcore,rgalaxy,rdist):
    """
    Function ellipticity calculates the ellipticity of each star's orbit.
    """
    #calculate ellipticity 
    #set up an array of ellipticities
    ellipt_array=np.empty(N)
    
    for i in range(N):
        #set up cases for what happens when a<rcore, rcore<a<rgalaxy, and 
        #rgalaxy<a<rdist
        
        if(a_vals[i]<rcore):
            ellipt_array[i]=1+((a_vals[i]/rcore)*(e1-1))
            
        elif(rcore<a_vals[i] and a_vals[i]<rgalaxy):
            ellipt_array[i]=e1+(((a_vals[i]-rcore)/(rgalaxy-rcore))*(e2-e1))
        
        elif(rgalaxy<a_vals[i] and a_vals[i]<rdist):
            ellipt_array[i]=e2+(((a_vals[i]-rgalaxy)/(rdist-rgalaxy))*(1-e2))
            
    return ellipt_array

def semi_minor_axis(N,a_vals,ellipt_array):
    """
    Function semi_minor_axis calculates the semi-minor axis, b, for each star.
    """            
    #calculate b values
    b_vals=np.empty(N)
    
    for i in range(N):
        b_vals[i]=ellipt_array[i]*a_vals[i]
    
    return b_vals

def tilts(N,a_vals):
    """
    Function tilts sets up the tilts of each star's orbit.
    """
    #calculate tilt of orbits
    #angular offset is in radians
    angular_offset=0.0004
    
    #set up a tilt array
    psi_angles=np.empty(N)
    for i in range(N):
        psi_angles[i]=-90+(a_vals[i]*angular_offset)
    
    return psi_angles

def positions(N,a_vals,b_vals,theta_angles,psi_angles):
    """
    Function positions returns the x and y coordinates of stars.
    """
    #setup arrays to hold x and y coordinates
    x_array=np.empty(N)
    y_array=np.empty(N)
    
    for i in range(N):
        #split equations into parts
        x_partone=a_vals[i]*np.cos(theta_angles[i])*np.cos(psi_angles[i])
        x_parttwo=b_vals[i]*np.sin(theta_angles[i])*np.sin(psi_angles[i])
        x_array[i]=x_partone-x_parttwo
        
        y_partone=a_vals[i]*np.cos(theta_angles[i])*np.sin(psi_angles[i])
        y_parttwo=b_vals[i]*np.sin(theta_angles[i])*np.cos(psi_angles[i])
        y_array[i]=y_partone+y_parttwo
 
    return x_array,y_array

def euler(value,timestep,vel):
    """
    Function euler performs Euler's method on an inputted value.
    """
    new_value=value+(vel*timestep)
    return new_value

"""
Main
"""
#a_stars is an array of each star's semi_major_axis
a_stars=semi_major_axis(Nstars)

#theta_stars is an array that holds the theta values for each star
#theta determines where the star is in its orbit
theta_stars=object_angle(Nstars)

#ellipt_stars is an array that holds the ellipticity values for each star
ellipt_stars=ellipticity(Nstars,a_stars,rcore,rgalaxy,rdist)

#b_stars is an array that holds the values for each star's semi-minor axis.
b_stars=semi_minor_axis(Nstars,a_stars,ellipt_stars)
    
#psi_stars stores the psi angles of each star, which determines the tilt.
psi_stars=tilts(Nstars,a_stars)

#x_stars and y_stars store the x- and y-coordinates
x_stars,y_stars = positions(Nstars,a_stars,b_stars,theta_stars,psi_stars)

#define value for number of evolutions
num_evols=int(input('Please input a value for the number of evolutions: '))

#define an array of updating thetas that update on every evolution
new_thetas=theta_stars

for i in range(num_evols):
    plt.plot(x_stars,y_stars,'o',markersize=0.2)
    for j in range(Nstars):
        #update theta values for new timestep
        new_thetas[j]=euler(new_thetas[j],timestep,velocity1)
    
    #get positions for star after a timestep
    x_stars,y_stars = positions(Nstars,a_stars,b_stars,new_thetas,psi_stars)
    
plt.show()