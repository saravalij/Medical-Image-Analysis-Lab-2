
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage.filters as filters
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import reconstruction, square, disk, cube
from skimage.segmentation import watershed



def coordinates(x,y,x_lim,y_lim):
    '''
    Retrieves the coordinates of the neighbour pixels to the introduced pixel
    coordinates, considering a connectivity of 8.

    Parameters
    ----------
    x : x-coordinate of the central pixel.
    y : y-coordinate of the central pixel.
    x_lim : Limit for the x, given by the shape of the image.
    y_lim : Limit for the y, given by the shape of the image.

    Returns
    -------
    coord_pixels: Coordinates of the neighbour pixels to the central pixel.

    '''
    
    coord_pixels=[]
    rest_x= [-1, -1 ,-1 ,0 ,0 ,1 ,1 ,1]
    rest_y= [-1 ,0, 1, -1, 1,  -1, 0, 1]
    for i in range(len(rest_x)):
        if (0 <= x-rest_x[i] < x_lim) and (0 <= y-rest_y[i] < y_lim):
            neigh_pix=[x-rest_x[i],y-rest_y[i]]
            coord_pixels.append(neigh_pix)

    return coord_pixels # array of coordinates arrays


def RegionGrowingP2(img, seed, gray_level):
    '''
    Performs the Region Growing Segmentation, based on assessing the homogenity
    between pixels.

    Parameters
    ----------
    img : Input image to be segmented.
    seed : Pixel within the ROI that we force our algorithm to start at.
    gray_level : Gray level threshold that defines the homogeneity condition.

    Returns
    -------
    rg_img : Segmented image.

    '''
    
    rg_img = np.zeros(img.shape) # Creating a black image
    
    x=int(round(seed[0][1],0)) # Seed x-coordinate
    y=int(round(seed[0][0],0)) # Seed y-coordinate
    seed1 = [x,y]
    
    roi=[seed1] # Stores the coordinates of those pixels that satisfy the homogenity condition and are therefore in the ROI
    pixel_evaluated=[seed1] # Stores the pixels that have already been added or not added to the roi, so that they are not repeated

    while len(roi)>0: # Assessing whether evaluated pixels are homogeneous to the seed
        coord_base=roi.pop() # Defines the pixel within the ROI that is being evaluated
        
        for j,k in coordinates(coord_base[0],coord_base[1],img.shape[1],img.shape[0]):
            
            if [j,k] not in pixel_evaluated:
                
                if (img[x][y]-gray_level)< img[j][k] < (img[x][y]+gray_level): #Homogeneity condition
                    roi.append([j,k]) # Appends homogeneous pixels to the ROI
                    rg_img[j][k]=1 # Sets homogeneous pixels to white (1)
                    
                pixel_evaluated.append([j,k]) # Once evaluated, append to the list
            
    
    return rg_img



def imimposemin(I, BW, conn=None, max_value=255):
    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J


def WatershedExerciseP2(img):
    '''
    Morphology-based segmentation technique that allows you to find regions of
    an image. For this we will calculate the maximum and minimum of the image 
    to be able to differentiate the edges of the background.
    
    Parameters
    ----------
    img: Input image to be segmented.
    
    Returns
    -------
    watershed(sobel_img): Resulting image after applying to the separate sobel image the algorithm.
    watershed(im_image): Image obtained with a better result after applying several seeds.
    watershed(sobel_img,im_image): Image showing only the edges of the segmented part. 

    '''
    
    sobel_img = filters.sobel(img) # Gradient image
    plt.figure()
    plt.imshow(img,cmap='gray')
    num_seed= 5
    pts = plt.ginput(num_seed)
    plt.axis('off')
    
    pts = np.array(pts)
    pts = pts.astype(int)
    binary_mask= np.zeros(img.shape) # Binary mask

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            for n in range(len(pts)):
                if (i ==pts[n][1] and j ==pts[n][0]):
                    binary_mask[i,j]=1
                        
    im_image = imimposemin(sobel_img, binary_mask)
    
    return watershed(sobel_img),watershed(im_image),watershed(sobel_img,im_image)

    
    
def activecontours(imagen,r,c,act_param):
    '''
    Finds a contour that best approximare to the boundaries of the object.
    
    Parameters
    ----------
    img: Input image to be segmented.
    r: Necessary parameters to delimit the part to be segmented (red line).
    c: Necessary parameters to delimit the part to be segmented (red line).
    act_param: List of three parameters necessary to delimit the part we want to segment (blue line).
        act_param[0]: alpha
        act_param[1]: beta
        act_param[2]: gamma
    
    Returns
    -------
    snake: Blue line resulting from applying the algorithm that segments the region of interest.
    int: Red line resulting from applying the algorithm that segments the region of interest.
    '''
    
    alpha,beta,gamma=act_param[0],act_param[1],act_param[2]
    
    s = np.linspace(0, 2*np.pi, 400)
    r = r[0] + r[1]*np.sin(s)
    c = c[0] + c[1]*np.cos(s)
    init = np.array([r, c]).T
    
    snake = active_contour(gaussian(imagen, 3, preserve_range=False),
                           init, alpha=alpha, beta=beta, gamma=gamma,coordinates='rc')
    return snake, init


def plot_img(images,titles,cmap=None):
    '''
    Plots multiple images.

    Parameters
    ----------
    images : Images to plot.
    titles : Title of the plots of each image.
    cmap : Color map of the plots.

    Returns
    -------
    None. The function is called and x plots are plotted.

    '''
    
    plt.figure(figsize=(11, 11))
    ind=1
    
    for i in images:
        plt.subplot(1,len(titles),ind)
        plt.imshow(i, cmap=cmap)
        plt.title(titles[ind-1]), plt.axis('off')
        ind+=1
    
    plt.tight_layout()
    plt.show()
     
    return None
        









