from scipy.special import j1
from scipy.signal import convolve2d
import numpy as np

# "airy disk" function (actually approximate, and a rescaling, but it's ok)
def airy_disk(x):   
    return np.power( 2.*j1(x)/(x) , 2)


# generate a (2*size+1,2*size+1) convolution kernel with "radii" scale
# where the function above is assumed to have "radius" one
# scale is a 3-vector for RGB
def generate_kernel(scale,size):
    x = np.arange(-size,size+1,1.0)
    y = np.arange(-size,size+1,1.0)

    xs, ys = np.meshgrid(x,y)

    kernel = np.zeros((xs.shape[0],xs.shape[1],3))

    r = np.sqrt(xs**2 + ys**2) + 0.000001
    kernel[:,:,:] = airy_disk(r[:,:,np.newaxis]/scale[np.newaxis,np.newaxis,:])

    #normalization
    kernel /= kernel.sum(axis=(0,1))[np.newaxis,np.newaxis,:]

    return kernel


# computed from approximate position of red green and blue in the spectrum
# it's a brutal approximation, but it vaguely looks like the real thing
SPECTRUM = np.array([1.,0.86,0.61])


# convolve a 2D RGB array with three airy kernels with radius of
# red channel = radius and the other two rescaled as of above
# the kernel pixel size is fixed by kernel_radius
def airy_convolve(array,radius,kernel_radius=25):
    kernel = generate_kernel(radius * SPECTRUM , kernel_radius)

    out = np.zeros((array.shape[0],array.shape[1],3))
    for i in range(3):
        out[:,:,i] = convolve2d(array[:,:,i],kernel[:,:,i],mode='same',boundary='symm')

    return out
