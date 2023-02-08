"""
    .. module:: pyir_image
       :platform: Windows
       :synopsis: Image object for certain module outputs and image registration actions.

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter
import skimage
from tkinter import ttk
from PIL import Image, ImageTk
from skimage import measure, segmentation
from sklearn.metrics.pairwise import euclidean_distances

from pyir_spectralcollection import *
from pyir_mask import * 
from pyir_pca import *


class PyIR_Image:
    """PyIR_Image is a class to be used mainly for imaging purposes such 
    as plotting and image registration of annotations.


    """
    def __init__(self, *args, **kwargs):
        
        self.xpixels = []
        self.ypixels = []
        super().__init__(*args, **kwargs)


    def greyscale(self, data):
        """Normalises the object's data variable to a greyscale [0, 255] range,
        plots the image using matplotlib's pyplot module, and returns the 
        greyscaled data array.
    
        :returns: matplotlib pyplot, numpy.array
        
        """  
    
        #Greyscaling
        #areas = areas - np.min(areas)
        areamax = np.max(data)
        areamin = np.min(data)
        
             
        data = ((data-areamin)/(areamax-areamin))*255
             
        return data
    
    def disp_image(self, image_data, title=False, colour_bar=False, greyscale = False,
                   ypixels=0, xpixels=0, **kwargs):    
        """Plots and displays the inputted image array using matplotlib's
        pyplot module. Additional arguments for title and colour bars 
        included.
        
        :param image_data: Image array.
        :type image_data: numpy.array or array.
        :param title: Title of the plot, default=False
        :type title: string.
        :param colour_bar: Colour bar toggle.
        :type colour_bar: bool.
        :param ypixels: number of y pixels in image.
        :type ypixels: int
        :param xpixels: number of x pixels in image.
        :type xpixels: int
        
        
        :returns: matplotlib.pyplot plot
        
        """
        #topleft = self.data[:, 0, 0]
        if ypixels ==0:
            ypixels = self.ypixels
        if xpixels ==0:
            xpixels = self.xpixels
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        if greyscale == True:
            im = ax1.imshow(image_data, cmap="gray")
        else:
            im = ax1.imshow(image_data)
        if colour_bar == True:
            cbar = plt.colorbar(im, ax=ax1)
        if title != False:
            plt.title(title)
        #cbar = plt.colorbar(im, ax=ax1)
        ax1.axis('off')
        plt.show()
        
        return im.figure
    
    def import_annotations(self, filepath, alpha = False):
        """Imports an annotation file using matplotlib.pyplot's imread 
        function.
        
        :param filepath: Filepath of the annotation.
        :type filepath: str
        
        :returns: array
        
        """
        image = plt.imread(filepath)
        
        if (alpha == False ) & (len(image[0,0,:]) == 4) :
            image = image[:,:,0:3]
        return image
    
    def rgb2gray(self, image):
        """Converts a RGB image to gray.
        
        :param image: RGB image array.
        :type image: array
        
        :returns: array
        
        """
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])*255
        
    def control_point_selection(self, annotations, source_image):
        """Produces two GUIs that are clickable images of a source
        and destination image. Left clicks on either image stores image 
        coordinates of the clicks into an array for usage in image 
        registration. Right clicks undo left click actions, removing the 
        coordinates from the array. Images must be a greyscale [y,x] array. In
        built functions automatically grayscale colour images.
        
        :param annotations: Image array of annotations.
        :type annotations: numpy.array
        :param source_image: Image array of source spectral scan.
        :type source_image: numpy.array
        
        :returns: object.source_coords as numpy.array, object.desti_coords as numpy.array
        
        """
        self.source_coords = []
        self.desti_coords = []
        #Idea is to use this function to select control points from which to 
        #calculate/estimate the transformation matrix needed in registerto
        
        root = tkinter.Toplevel()
        root.minsize(750, 500) 
        annotations = self.rgb_to_grey_check(annotations)
        img = ImageTk.PhotoImage(image=Image.fromarray(annotations))
        
        canvas = tkinter.Canvas(root, width=len(annotations[0,:]),
                                 height=len(annotations[:,0]))
        
        s_bar_vert = ttk.Scrollbar(canvas, orient="vertical")
        s_bar_horz = ttk.Scrollbar(canvas, orient="horizontal")
        
        s_bar_vert.config(command=canvas.yview)
        s_bar_horz.config(command=canvas.xview)

        canvas.config(yscrollcommand=s_bar_vert.set)
        canvas.config(xscrollcommand=s_bar_horz.set)
        
        s_bar_vert.pack(side='right', fill='y')
        s_bar_horz.pack(side='bottom', fill='x')
        canvas.pack(expand = 'yes', fill='both')
        canvas.config(scrollregion=(0,0,len(annotations[0,:])+20,
                                len(annotations[:,0])+20))
        canvas.create_image(0,0,anchor="nw",image=img)
        
        
        root2 = tkinter.Toplevel()
        root2.minsize(750, 500) 
        
        if source_image.ndim ==1:
            source_image = np.reshape(source_image,(self.ypixels,self.xpixels))
        img2 = ImageTk.PhotoImage(image=Image.fromarray(source_image))
        
        canvas2 = tkinter.Canvas(root2, width=len(source_image[0,:]),
                                 height=len(source_image[:,0]))
        
        s_bar_vert = ttk.Scrollbar(canvas2, orient="vertical")
        s_bar_horz = ttk.Scrollbar(canvas2, orient="horizontal")
        
        s_bar_vert.config(command=canvas2.yview)
        s_bar_horz.config(command=canvas2.xview)

        canvas2.config(yscrollcommand=s_bar_vert.set)
        canvas2.config(xscrollcommand=s_bar_horz.set)
        
        s_bar_vert.pack(side='right', fill='y')
        s_bar_horz.pack(side='bottom', fill='x')
        canvas2.pack(expand = 'yes', fill='both')
        canvas2.config(scrollregion=(0,0,len(source_image[0,:])+20,
                                len(source_image[:,0])+20))
        canvas2.create_image(0,0,anchor="nw",image=img2)
        
        def on_click_source(event):
            #noting the click coordinates from the destination image
            canvas = event.widget
            x_pos = canvas.canvasx(event.x)
            y_pos = canvas.canvasy(event.y)
            r = 5  # circle radius
            canvas2.create_oval(x_pos - r, y_pos - r, x_pos + r, y_pos + r, 
                               fill='red') 
            print("Source points added :", int(y_pos), int(x_pos))      
            self.source_coords.append([int(y_pos), int(x_pos)])
            
        def on_click_destination(event):
            #noting the click coordinates from the images
            canvas = event.widget
            x_pos = canvas.canvasx(event.x)
            y_pos = canvas.canvasy(event.y)
            r = 5  # circle radius
            canvas.create_oval(x_pos - r, y_pos - r, x_pos + r, y_pos + r, 
                               fill='red') 
            print("Destination points added :", int(y_pos), int(x_pos))      
            self.desti_coords.append([int(y_pos), int(x_pos)])
         
        def undo_click_source(event):
        #for undoing a previous point
            x_pos = self.source_coords[len(self.source_coords)-1][1]
            y_pos = self.source_coords[len(self.source_coords)-1][0]
            print("Removed Source Coordinates: ", y_pos, x_pos)      
            self.source_coords = self.source_coords[0:
                                                len(self.source_coords)-1]
                
        def undo_click_destination(event):
        #for undoing a previous point
            x_pos = self.desti_coords[len(self.desti_coords)-1][1]
            y_pos = self.desti_coords[len(self.desti_coords)-1][0]
            print("Removed Destination Coordinates: ", int(y_pos), int(x_pos))      
            self.desti_coords = self.desti_coords[0:
                                                len(self.desti_coords)-1]
        
        canvas.bind("<Button 1>",on_click_destination)
        canvas.bind("<Button 3>",undo_click_destination)
        canvas2.bind("<Button 1>",on_click_source)
        canvas2.bind("<Button 3>",undo_click_source)
        root.mainloop()    
        
        self.source_coords = np.array(self.source_coords)
        self.desti_coords = np.array(self.desti_coords)

    def rgb_to_grey_check(self, image_array):
        if image_array.ndim == 3:
                return self.rgb2gray(image_array)
        else:
                return image_array
            

    def click_to_coords(self, clickable_image, guide=None):
        """Produces one or two GUIs that are a clickable image and a second 
        un-interactable guide image. Left clicks on the clickable image stores 
        positional coordinates of the clicks into an array. Right clicks undo 
        left click actions, removing the coordinates from the array. 
        Images must be a greyscale [y,x] array. In built functions 
        automatically grayscale colour images.
        
        :param clickable_image: Image array of clickable image.
        :type clickable_image: numpy.array
        :param guide: Optional image array of guide image, defaul = None
        :type guide: numpy.array
        
        :returns: numpy.array
        
        """
        
        self.click_x_coords = []
        self.click_y_coords = []
        root = tkinter.Toplevel()
        root.minsize(750, 500) 
        
        if clickable_image.ndim == 1:
            clickable_image = np.reshape(clickable_image, (self.ypixels, self.xpixels))
        clickable_image = self.rgb_to_grey_check(clickable_image)
        img = ImageTk.PhotoImage(image=Image.fromarray(clickable_image))
        canvas = tkinter.Canvas(root, width=len(clickable_image[0,:]),
                                 height=len(clickable_image[:,0]))  
        
        s_bar_vert = ttk.Scrollbar(canvas, orient="vertical")
        s_bar_horz = ttk.Scrollbar(canvas, orient="horizontal")
        
        s_bar_vert.config(command=canvas.yview)
        s_bar_horz.config(command=canvas.xview)

        canvas.config(yscrollcommand=s_bar_vert.set)
        canvas.config(xscrollcommand=s_bar_horz.set)
        
        s_bar_vert.pack(side='right', fill='y')
        s_bar_horz.pack(side='bottom', fill='x')
        canvas.pack(side = 'left', expand = 'yes', fill='both')
        canvas.config(scrollregion=(0,0,len(clickable_image[0,:]),
                                len(clickable_image[:,0])))
        canvas.create_image(0,0,anchor="nw",image=img)

        def on_click_coord(event):
                #noting the click coordinates from the destination image
                canvas = event.widget
                x_pos = canvas.canvasx(event.x)
                y_pos = canvas.canvasy(event.y)
                r = 5  # circle radius
                r = 5  # circle radius
                canvas.create_oval(x_pos - r, y_pos - r, x_pos + r, y_pos + r, 
                                   fill='red') 
                self.click_y_coords.append([int(y_pos)])
                self.click_x_coords.append([int(x_pos)])
                
        canvas.bind("<Button 1>",on_click_coord)
        
        if str(type(guide)) != "<class 'NoneType'>":
            root2 = tkinter.Toplevel()
            root2.minsize(750, 500) 
            guide = self.rgb_to_grey_check(guide)
            img2 = ImageTk.PhotoImage(image=Image.fromarray(guide))     
            canvas2 = tkinter.Canvas(root2, width=len(guide[0,:]),
                                     height=len(guide[:,0]))
            s_bar_vert = ttk.Scrollbar(canvas2, orient="vertical")
            s_bar_horz = ttk.Scrollbar(canvas2, orient="horizontal")
            
            s_bar_vert.config(command=canvas2.yview)
            s_bar_horz.config(command=canvas2.xview)
    
            canvas.config(yscrollcommand=s_bar_vert.set)
            canvas.config(xscrollcommand=s_bar_horz.set)
            
            s_bar_vert.pack(side='right', fill='y')
            s_bar_horz.pack(side='bottom', fill='x')
            canvas2.pack(side = 'left', expand = 'yes', fill='both')
            
            canvas2.pack()
            canvas2.create_image(0,0,anchor="nw",image=img2)
        
        root.mainloop()
        
        return np.column_stack((self.click_y_coords,self.click_x_coords))     
        
    def crop_to_click(self, annotations, image):
        """Crops the annotations image array to a desired size determined
        by user determined coordinate array returned from click_to_coords().
        
        :param annotations: Image array of annotation image.
        :type annotations: numpy.array
        :param image: Optional image array of spectral scan image.
        :type image: numpy.array
        
        :returns: numpy.array
        
        """
        Coordinates = self.click_to_coords(annotations, image)
        leftmost = int(np.min(Coordinates[:,1]))
        rightmost = int(np.max(Coordinates[:,1]))
        top = int(np.max(Coordinates[:,0]))
        bot = int(np.min(Coordinates[:,0]))
        return annotations[bot:top,leftmost:rightmost]
    
    def match_size(self, annotations, source_image):
        """Reshapes the initial input array to the dimensions of the 
        second input, using skimage.transform module.
        
        :param annotations: Image array of annotation image.
        :type annotations: numpy.array
        :param image: Optional image array of spectral scan image.
        :type image: numpy.array
        
        :returns: numpy.array
        
        """
        return skimage.transform.resize(annotations, (len(source_image[:,0]), 
                                       len(source_image[0,:])))
    def resize_image(self, annotations, y_size=0, x_size = 0):
        """Reshapes the initial input array to the dimensions of the 
        inputs, using skimage.transform module.
        
        :param annotations: Image array of annotation image.
        :type annotations: numpy.array
        :param y_size: Desired y_pixel size. Default =0.
        :type y_size: int
        :param x_size: Desired x_pixel size. Default =0.
        :type x_size: int
        
        :returns: numpy.array
        
        """
        
        y_size = np.floor(y_size)
        x_size = np.floor(x_size)
        
        if y_size == 0:
            y_size = annotations.shape[0]
        if x_size == 0:
            x_size = annotations.shape[1]
        
        return skimage.transform.resize(annotations, (y_size, x_size))
    
    def affine_transform(self, annotations, source_coor, desti_coor):
        """Conducts an affine transform on inputted annotations using provided
        source and destination coordinates derived from the toolbox's
        control_point_selection() function.
        
        :param annotations: Image array of annotation image.
        :type annotations: numpy.array
        :param source_coor: Collection of [x,y] source coordinates.
        :type source_coor: numpy.array
        :param desti_coor: Collection of [x,y] destination coordinates.
        :type desti_coor: numpy.array
        
        :returns: numpy.array
        
        """
        
        affine_trans = skimage.transform.estimate_transform('affine', 
                                                    source_coor,desti_coor)
        
        warped = skimage.transform.warp(annotations, 
                                    inverse_map = affine_trans.inverse,
                         output_shape=(annotations.shape[0]+500, 
                                       annotations.shape[1]+500))
        return warped
    
    def fill_image_click(self, image_mask, value=1, **kwargs):
        """Fills user selected regions using skimage's segmentation tools.
        
        :param image_mask: Original image mask with veins
        :type image_mask: np.array
        :param value: Value to convert the regions into. Default=1
        :type value: int 
        
        :returns: numpy array
        
        """  
        if image_mask.ndim ==1:
            image_mask = np.reshape(image_mask, (self.ypixels, self.xpixels))
            
        coordinates = self.click_to_coords(image_mask)
        #coordinates are [y, x], flood_fill needs [y,x]
        
        if type(image_mask[0][0]) == np.bool_:
            is_bool = True
            image_mask = image_mask.astype(int)
            
        for i in np.arange(0, coordinates.shape[0]):
            image_mask = skimage.segmentation.flood_fill(image_mask, (coordinates[i,0],
                                                 coordinates[i,1]), value)

        return image_mask.astype(bool)
    
    def vein_finder(self, image_mask, value=1, threshold=10, expand=0, **kwargs):
        """Finds clusters within a boolean image to locate assumed vein 
        positions to be used for distancing metric work.
        
        :param image_mask: Original image mask with veins
        :type image_mask: np.array
        :param value: Value of the veins inside the image. Default=1
        :type value: int 
        :param threshold: Minimum pixel size of "veins". Default=10
        :type threshold: int 
        :param expand: Expand labels by set pixel amount to account for neighbouring ducts. Default=0
        :type expand: int 
        
        
        :returns: numpy array
        
        """  
        original_mask = (image_mask > 0)
        
        if image_mask.ndim ==1:
            image_mask = np.reshape(image_mask, (self.ypixels, self.xpixels))
         
        if type(image_mask[0][0]) == np.bool_:
           is_bool = True
           image_mask = image_mask.astype(int)
            
        if expand > 0:
            image_mask = segmentation.expand_labels(image_mask, distance=expand)
            
        image_mask =  measure.label(image_mask, background=0)
        
        #reshape and exclude segments under threshold
        image_mask =  np.reshape(image_mask,(self.ypixels*self.xpixels))
        for i in np.arange(1, np.max(image_mask)):
            if np.sum(image_mask==i) < threshold:
                image_mask[image_mask==i] = 0
                
        image_mask = measure.label(np.reshape(image_mask,(self.ypixels, 
                                                          self.xpixels)), 
                           background=0)
        
        if original_mask.ndim ==1:
            image_mask = (np.reshape(image_mask,(self.ypixels*self.xpixels))
                            * original_mask)
        elif original_mask.ndim == 2:
            image_mask = (np.reshape(image_mask,(self.ypixels*self.xpixels))
                        *np.reshape(original_mask,(self.ypixels*self.xpixels)))
        
        return image_mask
    
    def distance_mapper(self, vein_masks, tissue_mask, ypixels = None, xpixels = None, threshold=800):
        """Calculates the euclidean distance of tissue regions from the nearest
        detected veins within the tissue. Requires a numbered mask list of veins.
        see vein_finder() function.
        
        :param vein_masks: Original numbered mask of veins. Requirey array of int.
        :type vein_masks: np.array of int(32 or 64).
        :param tissue_mask: Mask of tissue regions.
        :type tissue_mask: np.array of int(32 or 64).
        :param ypixels: ypixels of the image. Default= None
        :type ypixels: int 
        :param xpixels: xpixels of the image. Default= None
        :type xpixels: int 
        :param threshold: Threshold for vein size to calculate distance from vein edge. Default = 800.
        :type threshold: int 
        
        
        :returns: pixel_distance, micron_distance (numpy arrays)
        
        """  
        
        if ypixels == None:
            ypixels = self.ypixels
        if xpixels == None:
            xpixels = self.xpixels
        
        distances_from_veins_all = np.zeros((ypixels*xpixels, np.max(vein_masks)))    
        
        positions_max = []
        for y in np.arange(0, ypixels):
            for x in np.arange(0, xpixels):
                positions_max.append([y,x])
        
        positions_max = np.array(np.array(positions_max))
        
        def findCOM(grid):
             Mx = 0
             My = 0
             mass = 0
            
             for i in np.arange(0, grid.shape[0]):
                Mx += grid[i,0]
                My += grid[i,1]
                mass += 1
             COM = np.array((Mx/mass, My/mass))
             return COM
        
        for z in np.arange(0, np.max(vein_masks)):
            #size check, if too big have to distance around the vein for best guess
            if np.sum(vein_masks == z+1) > threshold:
            #going to find the edge of the veins, and then just do distancing around it.
                edge = (segmentation.expand_labels(vein_masks == z+1, distance=1).astype(int)-
                        (vein_masks == z+1).astype(int)).astype(bool)
                around_edge_points = np.sum(edge).astype(int)
                distances_from_vein_big = np.zeros((ypixels*xpixels, 2))  
                edge = edge.astype(int)
                edge_count = np.arange(1, np.sum(edge)+1)
                edge[edge==1] = edge_count
                for k in np.arange(1, around_edge_points):   
                    vein_center = findCOM(np.array(positions_max)[edge == k])
                    distances = euclidean_distances(positions_max[tissue_mask], np.array(vein_center).reshape(1,-1)).ravel()
                    
                    if k == 1:
                        distances_from_vein_big[tissue_mask, 0] = distances
                    else:
                        distances_from_vein_big[tissue_mask, 1] = distances
                        distances_from_vein_big[tissue_mask, 0] = (
                            np.min(distances_from_vein_big[tissue_mask], axis=1))
                
                #if np.sum(distances_from_vein_big[:,around_edge_points-1]) == 0:
                    #distances_from_vein_big = distances_from_vein_big[:,:-1]
                distances_from_veins_all[tissue_mask, z] = distances_from_vein_big[tissue_mask,0]
            
            else:
                vein_center = findCOM(np.array(positions_max)[vein_masks == z+1])
                #avg_dist_from_cent =np.mean(np.abs(vein_center-
                #                                 np.array(positions_max)[vein_masks == z+1]))
                
                distances = euclidean_distances(positions_max[tissue_mask], np.array(vein_center).reshape(1,-1)).ravel()
                #distances = distances-avg_dist_from_cent
                
                distances_from_veins_all[tissue_mask, z] = distances
            
            print(z/np.max(vein_masks))
        
        pixels_distance = np.floor(np.reshape(np.min(
                distances_from_veins_all, axis=1),(-1,1)))
        
        micron_distance_cm = (pixels_distance*(5.5e-4))
        
        return pixels_distance, micron_distance_cm
    
    def tissue_excluder(self, tissue_mask, ypixels=None, xpixels=None):
        """Segments out all tissue debris or breaks that are not part of the
        main tissue sections. Accounts for multiple large sections by ensuring
        final mask contains >=95% of all tissue spectra.
        
        :param tissue_mask: Original mask of segmented tissues. Requires array of bool.
        :type tissue_mask: np.array of bool.
        :param ypixels: ypixels of the image. Default= None
        :type ypixels: int 
        :param xpixels: xpixels of the image. Default= None
        :type xpixels: int 
        
        :returns: numpy array of bool.
        
        """
        
        if ypixels == None:
            ypixels = self.ypixels
        if xpixels == None:
            xpixels = self.xpixels
        
        if tissue_mask.ndim == 1:
            numbered_sections = measure.label(np.reshape(tissue_mask,(ypixels,xpixels)), background=0)
        elif tissue_mask.ndim ==2:
            numbered_sections = measure.label(tissue_mask, background=0)
        
        size_counts = np.zeros((np.max(numbered_sections)))
        for n in np.arange(1,np.max(numbered_sections)):
            size_counts[n] = (np.sum(numbered_sections==n))
        
        tissue_frac = 0
        main_segment = np.zeros((ypixels, xpixels)).astype(bool)
        while tissue_frac <= 0.95:
            main_segment = main_segment +(numbered_sections == np.where(size_counts ==
                                                          np.max(size_counts))[0][0])
            size_counts[np.where(size_counts ==np.max(size_counts))[0][0]] = 0
            tissue_frac = np.sum(main_segment)/ np.sum(tissue_mask)

        return main_segment.ravel()
    
    def cluster_rebuild(self, clusters, mask, ypixels=None, xpixels=None):
        """Rebuilds a clustering output to an image.
        
        :param clusters: Original clusters results.
        :type clusters: np.array
        :param mask: Original tissue mask.
        :type mask: np.array of bools.
        :param ypixels: ypixels of the image. Default= None
        :type ypixels: int 
        :param xpixels: xpixels of the image. Default= None
        :type xpixels: int 
        
        :returns: numpy array of int.
        
        """
        
        if ypixels == None:
            ypixels = self.ypixels
        if xpixels == None:
            xpixels = self.xpixels
        
        rebuild_knn = np.zeros((ypixels*xpixels))
        count = 0;
        for i in np.arange(0, rebuild_knn.shape[0]):
            if mask[i] == True:
                rebuild_knn[i] = clusters[count]+1
                count = count+1
                
        return rebuild_knn.astype(int)
        