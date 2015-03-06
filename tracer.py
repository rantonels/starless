import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndim
import scipy.misc as spm
import random,sys,time
import datetime

INV255 = 1./255.

#defining texture lookup
def lookup(texarr,uvarrin): #uvarrin is an array of uv coordinates
    uvarr = np.clip(uvarrin,0.0,0.999)

    uvarr[:,0] *= float(texarr.shape[1])
    uvarr[:,1] *= float(texarr.shape[0])
    
    uvarr = uvarr.astype(int)

    return INV255*texarr[  uvarr[:,1], uvarr[:,0] ]


#rough option parsing
LOFI = False
SCENE_FNAME = 'scenes/default.scene'
for arg in sys.argv[1:]:
    if arg == '-d':
        LOFI = True
        continue
    if arg[0] == '-':
        print "unrecognized option: %s"%arg
        exit()
    
    SCENE_FNAME = arg
    


#importing scene
import ConfigParser

defaults = {
            "Distort":"1",
            "Fogdo":"1",
            "Blurdo":"1",
            "Fogmult":"0.1"         

            }

cfp = ConfigParser.ConfigParser(defaults)
print "Reading scene %s..."%SCENE_FNAME
cfp.read(SCENE_FNAME)

#some default values
#these should actually go in the "defaults" dict above
#but right now I don't feel like copying them.

RESOLUTION = (640,480)
NITER = 1000
STEP = 0.02

CAMERA_POS = np.array([0.,1.,-10.])

TANFOV = 0.8
LOOKAT = np.array([0.,0.,0.])

HORIZON_GRID = True

FOGDO = True
FOGMULT = 0.1
FOGSKIP = 1

BLURDO = True

SKYTEXFNAME = "textures/bgedit.png"

#this is never catched by except. WHY?
ex = ConfigParser.NoSectionError



#this section works, but only if the .scene file is good
#if there's anything wrong, it's a trainwreck
#must rewrite
try:
    RESOLUTION = map(lambda x:int(x),cfp.get('lofi','Resolution').split(','))
    NITER = int(cfp.get('lofi','Iterations'))
    STEP = float(cfp.get('lofi','Stepsize'))
except KeyError,ex:
    print "error reading scene file: insufficient data in lofi section"
    print "using defaults."


if not LOFI:
    try:
        RESOLUTION = map(lambda x:int(x),cfp.get('hifi','Resolution').split(','))
        NITER = int(cfp.get('hifi','Iterations'))
        STEP = float(cfp.get('hifi','Stepsize'))
    except KeyError,ex:
        print "no data in hifi section. Using lofi/defaults."

try:
    CAMERA_POS = map(lambda x:float(x),cfp.get('geometry','Cameraposition').split(','))
    TANFOV = float(cfp.get('geometry','Fieldofview'))
    LOOKAT = np.array(map(lambda x:float(x),cfp.get('geometry','Lookat').split(',')))
    UPVEC = np.array(map(lambda x:float(x),cfp.get('geometry','Upvector').split(',')))
    DISTORT = int(cfp.get('geometry','Distort'))

except KeyError:
    print "error reading scene file: insufficient data in geometry section"
    print "using defaults."


try:
    HORIZON_GRID = int(cfp.get('materials','Horizongrid'))
    DISK_TEXTURE = cfp.get('materials','Disktexture')
    SKY_TEXTURE = cfp.get('materials','Skytexture')
    SKYDISK_RATIO = float(cfp.get('materials','Skydiskratio'))
    FOGDO = int(cfp.get('materials','Fogdo'))
    BLURDO = int(cfp.get('materials','Blurdo'))
    FOGMULT = float(cfp.get('materials','Fogmult'))
except KeyError:
    print "error reading scene file: insufficient data in materials section"
    print "using defaults."


#just ensuring it's an np.array() and not a tuple/list
CAMERA_POS = np.array(CAMERA_POS)



print "Loading textures..."
if SKY_TEXTURE == 'texture':
    texarr_sky = spm.imread('textures/bgedit.png')
    if not LOFI:
        #   maybe doing this manually and then loading is better.
        print "(zooming sky texture...)"
        texarr_sky = spm.imresize(texarr_sky,2.0,interp='bicubic')

if DISK_TEXTURE == 'texture':
    texarr_disk = spm.imread('textures/adisk.jpg')
if DISK_TEXTURE == 'test':
    texarr_disk = spm.imread('textures/adisktest.jpg')


print "Computing rotation matrix..."
sys.stdout.flush()

# this is just standard CGI vector algebra

FRONTVEC = (LOOKAT-CAMERA_POS)
FRONTVEC = FRONTVEC / np.linalg.norm(FRONTVEC)

LEFTVEC = np.cross(UPVEC,FRONTVEC)
LEFTVEC = LEFTVEC/np.linalg.norm(LEFTVEC)

NUPVEC = np.cross(FRONTVEC,LEFTVEC)

viewMatrix = np.zeros((3,3))

viewMatrix[:,0] = LEFTVEC
viewMatrix[:,1] = NUPVEC
viewMatrix[:,2] = FRONTVEC


#array [0,1,2,...,numPixels]
pixelindices = np.arange(0,RESOLUTION[0]*RESOLUTION[1],1)

#total number of pixels
numPixels = pixelindices.shape[0]

#ones (useful for const->array conversion)
ones = np.ones((numPixels))
ones3 = np.ones((numPixels,3))

#random sample of floats
ransample = np.random.random_sample((numPixels))

def vec3a(vec): #returns a constant 3-vector array (don't use for varying vectors)
    return np.outer(ones,vec)

def vec3(x,y,z):
    return vec3a(np.array([x,y,z]))

def norm(vec):
    # you might not believe it, but this is the fastest way of doing this
    # there's a stackexchange answer about this
    return np.sqrt(np.einsum('...i,...i',vec,vec))

def normalize(vec):
    return vec/ (np.outer(norm(vec),np.array([1.,1.,1.])))

# an efficient way of computing the sixth power of r
# much faster than pow!
# np has this optimization for power(a,2)
# but not for power(a,3)!

def sqrnorm(vec):
    return np.einsum('...i,...i',vec,vec)

def sixth(v):
    tmp = sqrnorm(v)
    return tmp*tmp*tmp


# this blends colours ca and cb by placing ca in front of cb
def blendcolors(cb,balpha,ca,aalpha):
            #* np.outer(aalpha, np.array([1.,1.,1.])) + \
    return np.clip(
            ca +   
            cb * np.outer(balpha*(1.-aalpha),np.array([1.,1.,1.]))
                ,0.,1.)

# this is for the final alpha channel after blending
def blendalpha(balpha,aalpha):
    return aalpha + balpha*(1.-aalpha)


def saveToImg(arr,fname):
    print " - saving %s..."%fname
    #unflattening
    imgout = arr.reshape((RESOLUTION[1],RESOLUTION[0],3))
    #clip
    imgout = np.clip(imgout,0.0,1.0)
    plt.imsave(fname,imgout)

# this is not just for bool, also for floats (as grayscale)
def saveToImgBool(arr,fname):
    saveToImg(np.outer(arr,np.array([1.,1.,1.])),fname)



print "Generated %d pixel flattened array."%numPixels
sys.stdout.flush()

#arrays of integer pixel coordinates
x = pixelindices % RESOLUTION[0]
y = pixelindices / RESOLUTION[0]

print "Generating view vectors..."
sys.stdout.flush()


#the view vector in 3D space
view = np.zeros((numPixels,3))

view[:,0] = x.astype(float)/RESOLUTION[0] - .5
view[:,1] = ((-y.astype(float)/RESOLUTION[1] + .5)*RESOLUTION[1])/RESOLUTION[0] #(inverting y coordinate)
view[:,2] = 1.0

view[:,0]*=TANFOV
view[:,1]*=TANFOV

#rotating through the view matrix

view = np.einsum('jk,ik->ij',viewMatrix,view)

#original position
point = np.outer(ones, CAMERA_POS)

normview = normalize(view)

velocity = np.copy(normview)

#ignore all references to donemask. It's deprecated.
donemask = np.zeros((numPixels),dtype=np.bool)

# initializing the colour buffer
object_colour = np.zeros((numPixels,3))
object_alpha = np.zeros(numPixels)

#squared angular momentum per unit mass (in the "Newtonian fantasy")
h2 = np.outer(sqrnorm(np.cross(point,velocity)),np.array([1.,1.,1.]))

print "Starting pathtracing iterations..."
sys.stdout.flush()

start_time = time.time()

for it in range(NITER):
    if it%15 == 1:
        elapsed_time = time.time() - start_time
        progress = float(it)/NITER

        ETA = elapsed_time / progress * (1-progress)

        print "%d%%, %s remaining"%(int(100*progress), str(datetime.timedelta(seconds=ETA))),
        sys.stdout.flush()
        print "\r",

    #if it%100 == 1:
    #    saveToImg(object_colour,"tests/obcolseries/%d.png"%it)

    # STEPPING
    oldpoint = np.copy(point) #not needed for tracing. Useful for intersections

    #leapfrog method here feels good
    point += velocity * STEP
    
    if DISTORT:
        #this is the magical - 3/2 r^(-5) potential...
        accel = - 1.5 * h2 *  point / np.outer(sixth(point),np.array([1.,1.,1.]))
        velocity += accel * STEP


    #useful precalcs
    pointsqr = sqrnorm(point)
    phi = np.arctan2(point[:,0],point[:,2])
    theta = np.arctan2(point[:,1],norm(point[:,[0,2]]))
    normvel = normalize(velocity)


    # FOG

    if FOGDO and (it%FOGSKIP == 0):
        fogint = np.clip(FOGMULT * FOGSKIP * STEP / sqrnorm(point),0.0,1.0)
        fogcol = ones3
   
        object_colour = blendcolors(fogcol,fogint,object_colour,object_alpha)
        object_alpha = blendalpha(fogint, object_alpha)
        

    # CHECK COLLISIONS
    # accretion disk
    
    if DISK_TEXTURE != "none":

        mask_crossing = np.logical_xor( oldpoint[:,1] > 0., point[:,1] > 0.) #whether it just crossed the horizontal plane
        mask_distance = np.logical_and((pointsqr < 16.), (pointsqr > 2.25))  #whether it's close enough
        
        diskmask = np.logical_and(mask_crossing,mask_distance)

        if (diskmask.any()):

            if DISK_TEXTURE == "grid":
                diskcolor =     np.outer( 
                        np.mod(phi,0.52359) < 0.261799, 
                                    np.array([1.,1.,0.]) 
                                        ) +  \
                                np.outer(ones,np.array([0.,0.,1.]) )
                diskalpha = diskmask

            elif DISK_TEXTURE == "solid":
                diskcolor = np.array([1.,1.,.98]) 
                diskalpha = diskmask

            elif DISK_TEXTURE == "texture":
                uv = np.zeros((numPixels,2))
                
                uv[:,0] = (phi+np.pi)/(2*np.pi)
                uv[:,1] = (np.sqrt(pointsqr)-1.5)/(4.0-1.5)

                diskcolor = lookup ( texarr_disk, np.clip(uv,0.,1.))
                #alphamask = (2.0*ransample) < sqrnorm(diskcolor)
                #diskmask = np.logical_and(diskmask, alphamask )
                diskalpha = diskmask * np.clip(sqrnorm(diskcolor)/3.0,0.0,1.0)

            #object_colour += np.outer(np.logical_not(donemask)*diskmask,np.array([1.,1.,1.])) * diskcolor


            object_colour = blendcolors(diskcolor,diskalpha,object_colour,object_alpha) 
            object_alpha = blendalpha(diskalpha, object_alpha)

            donemask = np.logical_or(donemask ,  diskmask)
        

    # event horizon
    mask_horizon = np.logical_and((sqrnorm(point) < 1),(sqrnorm(oldpoint) > 1) )

    if mask_horizon.any() :

        if HORIZON_GRID:
            horizoncolour = np.outer( np.logical_xor(np.mod(phi,1.04719) < 0.52359,np.mod(theta,1.04719) < 0.52359), np.array([1.,0.,0.]))
        else:
            horizoncolour = np.outer(ones,np.array([0.,0.,0.]))#np.zeros((numPixels,3))
        
        #object_colour += np.outer(np.logical_not(donemask)*mask_horizon,np.array([1.,1.,1.])) * horizoncolour
        horizonalpha = mask_horizon  

        object_colour = blendcolors(horizoncolour,horizonalpha,object_colour,object_alpha) 
        object_alpha = blendalpha(horizonalpha, object_alpha)


        donemask = np.logical_or(donemask,mask_horizon)

print "Total raytracing time: %s"%(str(datetime.timedelta(seconds= (time.time()-start_time))) )

print "Computing final colour..."
sys.stdout.flush()

print "- generating sky layer..."
sys.stdout.flush()

vphi = np.arctan2(velocity[:,0],velocity[:,2])
vtheta = np.arctan2(velocity[:,1],norm(velocity[:,[0,2]]) )

vuv = np.zeros((numPixels,2))

vuv[:,0] = np.mod(vphi+4.5,2*np.pi)/(2*np.pi)
vuv[:,1] = (vtheta+np.pi/2)/(np.pi)

if SKY_TEXTURE == 'texture':
    col_sky = lookup(texarr_sky,vuv)[:,0:3] 

print "- generating debug layers..."
sys.stdout.flush()

#debug color: direction of view vector
dbg_viewvec = np.clip(view + vec3(.5,.5,0.0),0.0,1.0) 
#debug color: direction of final ray
dbg_finvec = np.clip(normalize(velocity) + vec3(.5,.5,0.0),0.0,1.0) 
#debug color: grid
dbg_grid = np.abs(normalize(velocity)) < 0.1
#debug color: donemask
dbg_done = np.outer(donemask,np.array([1.,1.,1.]))


if SKY_TEXTURE == 'texture':
    col_bg = col_sky
elif SKY_TEXTURE == 'none':
    col_bg = np.zeros((numPixels,3))
else:
    col_bg = dbg_finvec


print "- blending layers..."

#deprecated blend function
def blend(a1,a2,mask):
    mm = np.outer(mask,np.array([1.,1.,1.]))
    return np.logical_not(mm)*a1 + mm*a2

#col_bg_and_obj = blend(col_bg,object_colour,donemask)

col_bg_and_obj = blendcolors(SKYDISK_RATIO*col_bg, ones ,object_colour,object_alpha)


print "Postprocessing..."

#bloom

if BLURDO:
    hipass = np.outer(sqrnorm(col_bg_and_obj) > 2.0, np.array([1.,1.,1.])) * col_bg_and_obj
    blurd = np.copy(hipass)

    blurd = blurd.reshape((RESOLUTION[1],RESOLUTION[0],3))

    for i in range(3):
        print "- gaussian blur pass %d..."%i
        blurd = ndim.gaussian_filter(blurd,int(20./1024.*RESOLUTION[0]))

    blurd = blurd.reshape((numPixels,3))


    #final colour
    colour = np.clip(col_bg_and_obj + 0.3*blurd,0.,1.)#0.2*dbg_grid + 0.8*dbg_finvec

else:
    colour = col_bg_and_obj


print "Conversion to image and saving..."
sys.stdout.flush()

saveToImg(colour,"tests/out.png")
saveToImgBool(donemask,"tests/objects.png")
saveToImg(object_colour,"tests/objcolour.png")
saveToImgBool(object_alpha,"tests/objalpha.png")
saveToImg(col_bg_and_obj,"tests/preproc.png")
if BLURDO:
    saveToImg(hipass,"tests/hipass.png")


