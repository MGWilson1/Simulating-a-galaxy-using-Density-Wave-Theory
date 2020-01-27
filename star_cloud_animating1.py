"""
Project Title: Simulating a galaxy using Density Wave Theory.

star_cloud_animating1.py:  Animates stars and dust clouds orbitting and 
rotating around a galaxy by saving individual images and recalling them.  

Author: Michael Wilson
Student No.: 16364513
Date: 02/12/2019
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mgimg

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
#Nstars=number of stars in galaxy
Nstars=35000
#Nclouds=number of clouds
Nclouds=20000
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
    Function onject_angle calculates and returns the angle 'theta' for each
    star, cloud or astronomical object given.
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
    Function ellipticity calculates the ellipticity of each object's orbit.
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
    Function semi_minor_axis calculates the semi-minor axis, b, for each 
    object.
    """            
    #calculate b values
    b_vals=np.empty(N)
    
    for i in range(N):
        b_vals[i]=ellipt_array[i]*a_vals[i]
    
    return b_vals

def tilts(N,a_vals):
    """
    Function tilts sets up the tilts of each object's orbit.
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

def intensities(I0,rgalaxy,x,y):
    """
    Function intensities calculates the individual intensity of each star.
    """
    #calculate the distance from the centre of the galaxy=d
    d=np.sqrt((x**2)+(y**2))
    
    #calculate intensity for given star relative to intensity at centre of
    #galaxy
    intensity=I0*np.exp(-d/rgalaxy)
    
    return intensity

def euler(value,timestep,vel):
    """
    Function euler performs Euler's method on an inputted value.
    """
    new_value=value+(vel*timestep)
    return new_value

"""
Main
"""
#a_stars is an array of each star's orbit
#a_clouds is an array of each cloud's orbit
a_stars=semi_major_axis(Nstars)
a_clouds=semi_major_axis(Nclouds)

#create arrays to hold the theta angle values for the stars and clouds
theta_stars=object_angle(Nstars)
theta_clouds=object_angle(Nclouds)

#create arrays to hold ellipticity values for astronomical objects
ellipt_stars=ellipticity(Nstars,a_stars,rcore,rgalaxy,rdist)
ellipt_clouds=ellipticity(Nclouds,a_clouds,rcore,rgalaxy,rdist)

#create arrays to hold values for semi-minor axis for each object.
b_stars=semi_minor_axis(Nstars,a_stars,ellipt_stars)
b_clouds=semi_minor_axis(Nclouds,a_clouds,ellipt_clouds)
    
#create arrays to hold value for psi angles for each object
#the psi angle determines the tilt of the orbit
psi_stars=tilts(Nstars,a_stars)
psi_clouds=tilts(Nclouds,a_clouds)

#create arrays to hold positional values of each object
x_stars,y_stars = positions(Nstars,a_stars,b_stars,theta_stars,psi_stars)
x_clouds,y_clouds = positions(Nclouds,a_clouds,b_clouds,theta_clouds,\
                              psi_clouds)
    
#create arrays to hold the values for the brightness of each object as
#described by the brightness distribution across the galactic plane
star_intensities=intensities(I0,rgalaxy,x_stars,y_stars)
cloud_intensities=intensities(I0,rgalaxy,x_clouds,y_clouds)

#define value for number of evolutions
num_evols=int(input('Please input a value for the number of evolutions: '))

#create separate colour maps for stars and clouds
#cmap1 is for stars, cmap2 is for clouds
cmap1='Oranges_r'
cmap2='Blues_r'

#set up a for loop that saves every image to its own individual image file
#to get next image, use Euler's method to update theta and psi
for i in range(num_evols):
    fig=plt.figure(i+1)
    #set colours to match the intensities calculated
    c1=star_intensities
    c2=cloud_intensities
    plt.scatter(x_stars, y_stars, c=c1,s=1,cmap=cmap1,alpha=1)
    plt.scatter(x_clouds, y_clouds, c=c2,s=1000,cmap=cmap2,alpha=0.01)
    plt.xlim([-25000,25000])
    plt.ylim([-25000,25000])
    plt.savefig('figure_'+str(i)+'.png')

    #####################################
    #update values due to passing of time
    theta_stars=euler(theta_stars,timestep,velocity1)
    psi_stars=euler(psi_stars,timestep,velocity2)
    
    theta_clouds=euler(theta_clouds,timestep,velocity1)
    psi_clouds=euler(psi_clouds,timestep,velocity2)
    
    x_stars,y_stars = positions(Nstars,a_stars,b_stars,theta_stars,psi_stars)
    x_clouds,y_clouds = positions(Nclouds,a_clouds,b_clouds,theta_clouds,\
                                  psi_clouds)
    
    #recalculate intensities of objects based off their new positions
    star_intensities=intensities(I0,rgalaxy,x_stars,y_stars)
    cloud_intensities=intensities(I0,rgalaxy,x_clouds,y_clouds)

#set an initial object
fig=plt.figure(1)
ax=plt.gca()

imobj=ax.imshow(np.zeros((10,10)),origin='lower')

# initialization function 
def init(): 
    imobj.set_data(np.zeros((num_evols,num_evols)))
    return imobj,

# animation function 
def animate1(i): 
    #read in image and send it to initial object for animation
    img=mgimg.imread('figure_'+str(i)+'.png')[-1::-1]
    imobj.set_data(img)
    return imobj,
    
anim1=FuncAnimation(fig,animate1,frames=num_evols,interval=200,\
                blit=True,repeat=True)

plt.show()

mp4_title=input('Please provide a suitable name for the .mp4 file: ')
anim1.save(mp4_title,fps=10)
