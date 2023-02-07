"""
    .. module:: pyir_spectralcollection
       :platform: Windows
       :synopsis: This is the spectral collection class for groups of spectral data.

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import sys
import os

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\agilent-ir-formats\agilentirformats\agilentirformats'
sys.path.append(full_path)

import numpy as np
import matplotlib.pyplot as plt
import agilentmosaic as agmos
import agilenttile as agmostile
import scipy
import scipy.sparse
import scipy.integrate as sci_int
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split

import pyir_image, pyir_mask, pyir_pca

class PyIR_SpectralCollection:
    """This is the main Class for loading and preprocessing the spectral data.
    It also contains plotting functions, data handling, and reshaping tools. 

    PyIR_SpectralCollection() initialiser initialises some useful variables as 
    empty arrays for later usage.

    Kwargs:
       filepath (str): This is the filepath of the FT-IR tile or mosaic
       from which the data shall be loaded. Default = 'Not Specified'.
       
       file_type(str): This is an optional argument to pass what type of 
       data, tile or mosaic, is being loaded. There is a function for 
       determining the file type automatically. Default = 
       'Not Specified'.

    """
    
    def __init__(self, filepath = 'Not Specified', file_type = 'Not Specified'):
        
        self.file_type = file_type
        self.data = []
        self.data_loaded = False
        self.wavenumbers = []
        self.totalspectrum = []
        self.totalimage = []
        self.filedelim = '.'
        self.filepath = filepath
        self.path_exists = []
        self.file_readable = []
        self.xpixels = []
        self.ypixels = []
        self.source_coords = []
        self.desti_coords = []
        self.trans_source_coor = []
        self.trans_desti_coor = []

    def get_file_location(self):
        """This function simply prints the filepath of the file being loaded.
        """
        print(self.filepath)
        
    def set_file_location(self):
        """Sets file location to a user defined filepath different from that
        stated in the __init__().

        :param input: Filepath.
        :type name: str.
        :returns:  object.filepath set to filepath.

        """
        self.filepath = input('Set File Location: ')
        self.filepath = os.path.join(self.filedelim,self.filepath)
        
    def set_file_type(self):
         """Sets file type to a user define type of either 'tile' or 'mosaic'.
         
         :param input: File type: 'tile' or 'mosaic'.
         :type name: str.
         :returns:  object.file_type set to input.

         """
         self.file_type = input('Set File Type: ')
        
    def filepath_exists(self):
        """Checks whether filepath supplied is a valid filepath.
        
        :returns: bool
        
        """
        self.path_exists = os.path.exists(self.filepath)
        if self.path_exists == True:
            print("Filepath is valid")
        else:
            print("Filepath is invalid")
    
    def determine_filetype(self):
        """Detects the file type of the file supplied in __init__() and sets
        item.file_type as detected file type.

         """
        if self.filepath.lower() == "not specified":
            print("Filepath is invalid")
            self.filepath = input('Set File Location: ')
        else:
            if self.filepath[-4:].lower() not in (".dmd", ".drd", ".dms",
                                                  ".dmt"):
                self.file_type = 'tile'
            else:
                self.file_type = 'mosaic'
    
    def set_reader_module(self):
        """Sets the external agilent_ir_format module for reading in data 
        depending on file type. Uses determine_filetype() func.
        
        :returns:  agilent_ir_format module.

        """
        self.determine_filetype()
        temp = []
        if self.file_type.lower() == 'mosaic':
            temp = agmos.AgilentMosaic()
        elif self.file_type.lower() == 'tile':
            temp = agmostile.AgilentTile()   
        return temp
            
    def is_readable(self):
        """Checks whether file supplied to the toolbox can be opened by 
        external agilent_ir_format module.
        
        :returns: bool
        
        """
        self.determine_filetype()
        temp = self.set_reader_module()
        self.file_readable = temp.isreadable(self.filepath)
        
        if self.file_readable == True:
            print("File is readable")
        else:
            print("File is unreadable")
        
    def load_data(self):
        """Loads the data using external agilent_ir_format module.
        Data is loaded in as wavenumbers by x by y numpy arrayformat. It is 
        then reshaped internally to be of form x by y by wavenumber numpy array.
        
        :returns: object.items; data, wavenumbers, is_loaded, xpixels, ypixels
        
        """
        self.determine_filetype()       
        temp = self.set_reader_module()
        result = temp.read(self.filepath)
        print("info : ")
        print(result['info'])
        self.data = result['ydata']
        #Data is setup as wave by y by x, reshape to y by x by wave
        self.data = np.moveaxis(self.data,0,-1)
        self.wavenumbers = result['xdata']
        self.data_loaded = True
        self.xpixels = result['info']['xpixels']
        self.ypixels = result['info']['ypixels']
        _, self.data = self.reshaper_3D(self.data)
    
    def is_loaded(self):
        """Checks whether the data is loaded.
        
        :returns: bool
        
        """
        if self.data_loaded == False:
            print("Data is not loaded, attempting to load...")
            self.load_data()
            self.data_loaded = True
        else:
            print("Data is loaded")
   
    def update_sums(self):
        """Updates the item.totalimage array as a sum of all absorbances
        over all wavenumber absorbances and item.totalspectrum as a sum 
        of all wavenumbers.
        
        :returns: object.totalimage as numpy.array, object.totalspectrum as numpy.array
        
        """
        if self.data.ndim == 2:
            self.totalspectrum = np.sum(self.data, axis=0)
            self.totalimage = np.sum(self.data, axis=1)
            self.totalimage = np.reshape(self.totalimage, (self.ypixels,
                                                           self.xpixels))
        elif self.data.ndim ==3:
            self.totalspectrum = np.sum(np.sum(self.data, 0), 1)
            self.totalimage = np.sum(self.data, 2)
    
    def append(self, data1, data2, *args, axis=0):
        """Appends two spectral datasets in to one dataset along axis.
         
        :param data1: Numpy array of spectral dataset
        :type data1: numpy.array
        :param data2: Numpy array of spectral dataset
        :type data2: numpy.array
        :param args: Additional spectral datasets
        :type args: numpy.array
        :param axis: Axis which to append datasets along, default = 0
        :type axis: int
         
        :returns: numpy.array
        
        """
        output = np.append(data1, data2, axis = axis)
        for arg in args:
            output = np.append(output, arg, axis = axis)
        
        return output
        
    
    def keep_range(self, upper = 3500, lower = 1000, data=0, wavenums=0):
        """Keep range clips the spectral dataset to the specified wavenumber
        range. Data and wavenumber inputs can either be chosen, or default to
        the objects .data and .wavenumbers parameters.
        
        :param upper: Upper wavenumber window point, default = 3500.
        :type upper: int
        :param lower: Lower wavenumber window point, default = 1000.
        :type lower: int
        :param data: Dataset to be ammended. default =0.
        :type data: numpy.array
        :param wavenums: Lower wavenumber window point, default = 0.
        :type wavenums: numpy.array
        :returns: data as numpy.array, wavenums as numpy.array
        
        """
        upper, lower = self.arg_size_checker(upper, lower)
        if type(data) == int:
            data = self.data
        if type(wavenums) == int:
            wavenums = self.wavenumbers
            
        lowerlim = np.where(wavenums >= lower)[0][0]
        upperlim = np.where(wavenums >= upper)[0][0]
        #upperlim+1 and lowerlim-1 is to ensure that the upper and lower
        #limit column is also included
        
        if data.ndim == 3:
            data = data[:,:,lowerlim-1:upperlim+1]
        elif data.ndim ==2:
           data = data[:,lowerlim-1:upperlim+1]
        
        wavenums = wavenums[lowerlim-1:upperlim+1]
        
        return data, wavenums

        
    def reshaper_3D(self, data, reshapedim = []):
        """Reshaper_3D is dual functional: Inputting a 3-D dataset without any
        additional arguments converts the 3-dimensional spectral dataset 
        [n, m, wavenumber] to a 2-dimensional dataset [n*m, wavenumber] 
        while noting the dimensions of the original dataset [height, width, 
        wavenumber length].
        Inputting a 2-D dataset without any additional arguments converts the
        2-dimensional dataset [n*m by wavenumber] to a 3-dimensional dataset
        [n, m, wavenumber] where n and m are determined by the object's
        ypixel and xpixel values. Providing a dimensional vector 
        [n, m, wavenumber] reshapes the 2-dimensional dataset to those 
        dimensions.
        
        :param data: Dataset to be reshaped [n,m,wavenum] or [n*m, wavenum]
        :type data: numpy.array
        :param reshapeddim: Dimensions to reshape dataset [height, width, 
                                                           wavenum]
        :type reshapeddim: list of int.
        :returns: dims as list, data as numpy.array 
        
        """
    #First usage would determine the object is 3D and reshape.
    #Calling a second time will reshaped from 2D back to original 3D object
        if(data.ndim == 3):
            dim_1 = len(data[:,0,0])
            dim_2 = len(data[0,:,0])
            dim_3 = len(data[0,0,:])
            data = np.reshape(data,((dim_1*dim_2),dim_3))
            return [dim_1, dim_2, dim_3], data
        elif(data.ndim == 2 and len(reshapedim) == 3):
            data = np.reshape(data,(reshapedim[0],reshapedim[1],
                                                         reshapedim[2]))
        return data
            
    def reshaper_2D(self, data, reshapedim=[]):
        """Reshaper_2D is dual functional: Inputting a 2-D dataset without any
        additional arguments converts the 2-dimensional spectral dataset 
        [n, m] to a 1 dimensional dataset [1, n*m] while noting the dimensions
        of the original dataset [n, m]. Inputting a 1-D dataset without any 
        additional arguments converts the 1-dimensional dataset [1, n*m] to 
        a-3 dimensional dataset [n, m] where n and m are determined by the 
        object's ypixel and xpixel values. Providing a dimensional vector 
        [n, m] reshapes the 1-dimensional dataset to those dimensions.
        
        :param data: Dataset to be reshaped [n,m] or [1, n*m]
        :type data: numpy.array
        :param reshapedim: Dimensions to reshape dataset [height, width]
        :type reshapedim: list of int.
        :returns: dims as list, data as numpy.array
        
        """
        #for reshaping an array from nbym to 1by(n*m) and then back again
        #if necessary
        if(data.ndim == 2):
            dim_1 = len(data[:,0])
            dim_2 = len(data[0,:])
            data = np.reshape(data, (dim_1*dim_2))
            return [dim_1, dim_2], data
        elif(data.ndim == 1 and len(reshapedim) == 2):
            data = np.reshape(data,(reshapedim[0],reshapedim[1]))
        elif(data.ndim==1 and reshapedim == []):
            data = np.reshape(data,(self.ypixels,self.xpixels)) 
            
        return data
            
    def disp_spectra(self, data, wavenums, **kwargs):   
        """Plots and displays the inputted spectral absorbances using 
        matplotlib's pyplot module.
        
        :param data: Spectral absorbance array.
        :type data: numpy.array or array.
        :param wavenums: Spectral wavenumber array.
        :type wavenums: numpy.array or array.
        
        :returns: matplotlib.pyplot plot
        
        """     
        if data.ndim > 1:
            data = data.T
        
        fig = plt.figure()
        plt.plot(wavenums, data, **kwargs)
        plt.ylabel('Absorbance')
        plt.xlabel('Wavenumber cm^-1')  
        plt.show()
        
        return fig
        
    def disp_conf_int(self, data, wavenums=0, labels=0, shade=True ,title=0, **kwargs):
        """Plots and displays the inputted spectral absorbances with shaded
        confidence interval using matplotlib's pyplot module.
        Assumed that distributions are normal and 90% confidence interval
        
        :param data: Spectral absorbance array.
        :type data: numpy.array or array.
        :param wavenums: Spectral wavenumber array. Default = 0
        :type wavenums: numpy.array or array.
        :param labels: Optional labels object for plotting. Default = 0
        :type labels: Doug_ToolBox.make_class_labels object [names, num of spec]
        :param title: Optional argument for plot title. Default = 0
        :type labels: string
        
        :returns: matplotlib.pyplot plot
        
        """    
        
        if type(wavenums) == int:
            wavenums = np.arange(0, data.shape[1])
        
        if type(labels) != int:
            item_count = labels[0].shape[0]
            std = {}
            data_group_means = {}
            for i in np.arange(0, item_count):
                data_section = data[labels[1].ravel()==i]
                data_group_means[i] = np.mean(data_section, axis=0)
                std[i] = np.std(data_section, axis=0)
        else:
            item_count = 1
            std = {}
            data_group_means = {}
            for i in np.arange(0, item_count):
                data_group_means[i] = np.mean(data, axis=0)
                std[i] = np.std(data, axis=0)
        
            
        fig = plt.figure()
        if type(labels) != int:
            for i in np.arange(0, item_count):
                plt.plot(wavenums, data_group_means[i], label = labels[0][i][0])
                if shade==True:
                    plt.fill_between(wavenums,
                                     data_group_means[i]-(1.645*std[0][i]),
                                     data_group_means[i]+(1.645*std[0][i]),
                           alpha = 0.25       
                        )
                plt.legend()
            plt.ylabel('Absorbance')
            plt.xlabel('Wavenumber cm^-1')
        else:
            plt.plot(wavenums, data_group_means[0], label = "Group Mean")
            if shade==True:
                plt.fill_between(wavenums,
                                 data_group_means[0]-(1.645*std[0][i]),
                                 data_group_means[0]+(1.645*std[0][i]),
                       alpha = 0.25       
                    )
            plt.legend()
            plt.ylabel('Absorbance')
            plt.xlabel('Wavenumber cm^-1')  
        if title !=0:
            plt.title(str(title))
        plt.show()
        
        return fig
        
    def disp_distr(self, data, bins='auto', **kwargs):   
        """Plots and displays the distribution of an array of values.
        
        :param data: Value array.
        :type data: numpy.array or array.
        :param bins: Number of Bins. Default is 'auto'
        :type bins: 'auto' or int.
        :returns: matplotlib.pyplot plot
        
        """
        
        if data.ndim == 2:
            data, _ = self.reshaper_2D(data)
            
        plt.figure()
        plt.hist(data, bins = bins, **kwargs)
        plt.ylabel('Counts')
        plt.xlabel('Value')
        plt.show()
        
        
    def data_deriv(self, data, wavenumbers, window_length, poly_order, deriv):
        """Calculates the nth order derivative of the spectral dataset using
        the Savitzky Golay method.
        
        :param data: Spectral absorbance array.
        :type data: numpy.array or array.
        :param wavenumbers: Wavenumber array.
        :type wavenumbers: numpy.array or array.
        :param window_length: Window length.
        :type window_length: int
        :param poly_order: Polynomial order of fitted response.
        :type poly_order: int
        :param deriv: Derivative order.
        :type deriv: int
        :returns: wavenumbers numpy.array, deriv_data numpy,array
        
        """     
        
        data = scipy.signal.savgol_filter(data, axis=1, 
                    window_length = window_length, polyorder=poly_order,
                    deriv = deriv)
        
        deletion_point = int((window_length-1)/2)
        data = data[:,(deletion_point-1):(len(data[0,:])-deletion_point-1)]
        wavenumbers = wavenumbers[(deletion_point-1):
                                       (len(wavenumbers[:])
                                        -deletion_point-1)]
        
        
        return data, wavenumbers   
    
    def min2zero(self, data):
        """Adjusts spectral dataset to bring dataset min to zero.

        :param data: Spectral absorbance array.
        :type data: numpy.array or array.

        :returns: object.data
        
        """  
        min_test = np.min(data)
        data = data - min_test
        return data
        
    def all_spec_min2zero(self, data):
        """Adjusts spectral dataset to bring each spectra min to zero.
        
        :param data: Spectral absorbance array.
        :type data: numpy.array or array.
        
        :returns: object.data
        
        """  
        is_3D = False
        if data.ndim ==3:
            dims, data = self.reshaper_3D(data)
            is_3D = True
        
        mins = np.min(data, axis=1)
        data = data - mins[:,None]
            
        if is_3D ==True:
            data = self.reshaper_3D(data, dims)
        return data
        
        
    def mean_center(self, data):
        """Mean centers the inputted dataset.
        
        :param data: Spectral absorbance array.
        :type data: numpy.array or array.

        :returns: numpy.array
        
        """  
        
        if data.ndim ==3:
            _, data = self.reshaper_3D(data)
           
        mean = np.mean(data, axis = 0, keepdims = True)
        
        return data - mean  
        
        

    def vector_norm(self, data=0):
        """Vector normalises the object's spectral dataset along each 
        wavenumber measurement point.
        
        :param data: dataset to vector normalise. Default = 0
        :type data: array of float64.

        :returns: numpy.array
        
        """  
        #INITIAL CODE TAKEN FROM ALEX HENDERSON'S VECTORNORM FROM MATLAB
        if type(data) == int:
            data = self.data
        
        is_3D = False
        if data.ndim ==3:
            dims, data = self.reshaper_3D(data)
            is_3D = True
        
        is_sparseinput = scipy.sparse.issparse(data)
        
        squares = np.square(data)   # square of each variable ([n,m])
        sum_of_squares = np.sum(squares, axis = 1)   #sum of the squares along the rows ([1,n]) 
        divisor = np.sqrt(sum_of_squares)    # ([n,1]) == L2-Norm
        divisor[divisor == 0] = 1 #avoid div/0 error
        #Make data sparse matrix for memory 
        
        # Generate a sparse matrix where the diagonal is of the inverse of the divisor
        multiplier = 1./ divisor
        
        #delete items for memory 
        del squares
        del sum_of_squares
        del divisor
        
        multiplierdiag = scipy.sparse.csr_matrix(scipy.sparse.diags(multiplier))
        
        # multiply the two matrices and then convert data back to a dense matrix
        # which is dividing the data by the vector length 
        data = multiplierdiag * data
        #self.data = self.data.todense()
        
        del multiplierdiag
        #self.data = np.matmul(self.data,multiplierdiag) # divide the data by the vector length ([n,m])
        
        if is_sparseinput == True:
            data = scipy.sparse.csr_matrix(data)
        
        if is_3D == True:
            data = self.reshaper_3D(data, dims)
        
        return data
    
    def feature_norm(self, data=0, feature_pos=1650, wavenumbers=0):
        """ Featur normalises the object's spectral dataset along each 
        wavenumber measurement point based on a specified absorbance point.
        
        :param data: dataset to vector normalise. Default = 0
        :type data: array of float64.
        :param feature_pos: wavenumber position to normalise to. Default = 1650
        :type amideI: double/int.
        :param wavenumbers: wavenumber vector to find position. Default =0
        :type wavenumbers: array of float.

        :returns: numpy.array
        
        """  
        #INITIAL CODE TAKEN FROM ALEX HENDERSON'S VECTORNORM FROM MATLAB
        if type(data) == int:
            data = self.data
        
        is_3D = False
        if data.ndim ==3:
            dims, data = self.reshaper_3D(data)
            is_3D = True
        
        is_sparseinput = scipy.sparse.issparse(data)
        
        #To find the squares, just use internal .area_at function 
        squares = self.area_at(feature_pos,data,wavenumbers)
        sum_of_squares = squares
        divisor = sum_of_squares
        divisor[divisor == 0] = 1 #avoid div/0 error
        
        # Generate a sparse matrix where the diagonal is of the inverse of the divisor
        multiplier = 1./ divisor
        
        #delete items for memory 
        del squares
        del sum_of_squares
        del divisor
        
        multiplierdiag = scipy.sparse.csr_matrix(scipy.sparse.diags(multiplier))
        
        # multiply the two matrices and then convert data back to a dense matrix
        # which is dividing the data by the vector length 
        data = multiplierdiag * data
        #self.data = self.data.todense()
        
        del multiplierdiag
        #self.data = np.matmul(self.data,multiplierdiag) # divide the data by the vector length ([n,m])
        
        if is_sparseinput == True:
            data = scipy.sparse.csr_matrix(data)
        
        if is_3D == True:
            data = self.reshaper_3D(data, dims)
        
        return data
            
    def remove_range(self, upper, lower, data, wavenumbers):
        """Replaces the absorbance data between two wavenumber limits with a 
        straight line. Removing the data in that window from the dataset.
        
        :param upper: Upper wavenumber limit.
        :type upper: int.
        :param lower: Lower wavenumber limit
        :type lower: int.
        :param data: Data to apply function to.
        :type data: np.array
        :param wavenumbers: Wavenumber array.
        :type wavenumbers: np.array

        :returns: object.data
        
        """  
        upper, lower = self.arg_size_checker(upper, lower)
        
        lowerlim = np.where(wavenumbers >= lower)[0][0]  
        upperlim = np.where(wavenumbers >= upper)[0][0]
        #To save on computational time apply the removal depending on what
        #Form the data is in, 3D or 2D
        if data.ndim == 2:              
                data = np.concatenate((data[:,0:lowerlim], data[:,upperlim:]),
                                      axis=1)
                wavenumbers = np.concatenate((wavenumbers[0:lowerlim], 
                                              wavenumbers[upperlim:]))
        elif data.ndim ==3:
                data = np.concatenate((data[:,:,0:lowerlim], 
                                       data[:,:,upperlim:]),
                                      axis=1)
                wavenumbers = np.concatenate((wavenumbers[0:lowerlim], 
                                              wavenumbers[upperlim:]))
        return data, wavenumbers
            
    def remove_co2(self, data, wavenumbers):
        """Uses the toolbox's remove_range function to remove the CO2
        absorbance region found between 2450-2250 cm-1.
        
        """  
        data, wavenumbers = self.remove_range(upper = 2450, lower = 2250, data=data, 
                          wavenumbers=wavenumbers)
        
        return data, wavenumbers
        
    def remove_wax(self, data, wavenumbers):
        """Uses the toolbox's remove_range function to remove the♦ wax
        absorbance regions found between 3000-2700, 2400-2300, and
        1490-1350cm-1.
        
        """ 
        data, wavenumbers = self.remove_range(upper = 1490, lower = 1340, data=data, 
                          wavenumbers=wavenumbers)
        data, wavenumbers = self.remove_range(upper = 2400, lower = 2300, data=data, 
                          wavenumbers=wavenumbers)
        data, wavenumbers = self.remove_range(upper = 3000, lower = 2700, data=data, 
                          wavenumbers=wavenumbers)
        
        return data, wavenumbers
    
    def make_class_labels(self, label_array, num_of_spec_array):
        """Creates a class label array and list of labels based on input
        arguments.
        
        :param label_array: List of label names.
        :type upper: list of str.
        :param lower: List of spectra counts for each label.
        :type lower: list of int.

        :returns: label_names as numpy.array, class_labels as numpy.array
        
        """
        #form of labels must be ['label1', 'label2', ...] and
        # [num1, num2, .....]
        temp_pos = 0
        
        label_array = np.array(label_array)
        num_of_spec_array = np.array(num_of_spec_array)
        
        class_label_names = np.empty([len(label_array),1],
                                          dtype=label_array.dtype)
        class_labels = np.empty([sum(num_of_spec_array),1], dtype=int)
        
        for i in range(0, len(label_array)):
            class_label_names[i,0] = label_array[i][:]
            class_labels[temp_pos:(temp_pos+
                              num_of_spec_array[i]),0] = i
            temp_pos = (temp_pos+ num_of_spec_array[i])
  
        return [class_label_names, class_labels]
    
    def set_class_labels(self, label_array, num_of_spec_array):
        """Uses toolbox's make_class_labels function to set the object's
        internal class label items.
        
        """
        labels = self.make_class_labels(label_array, num_of_spec_array)
        self.class_label_names = labels[0]
        self.class_labels = labels[1]
        
        
    def K_means_cluster(self, data, clusters=5, init_ ='k-means++', n_initi =15,
                      max_it=300, tolerance= 0.01, **kwargs):
        #standardise the totalimage for clustering
        #converting the total image to type float is to conserve numerical 
        #accuracy
        data = preprocessing.scale(data.astype(float))
        
        if data.ndim == 3:
            data = self.reshaper_3D(data)
                
        k_means = KMeans(n_clusters=clusters, init=init_, n_init=n_initi,
                         max_iter=max_it, tol=tolerance, **kwargs).fit(data) 

        return k_means  
        
    
    def spectrum_at(self, data, wavenumbers, spectrum_loc):
        """Extracts the absorbance spectrum at a specified location, either 
        as a [y,x] position or an nth spectrum. Function assumes user 
        starts array at 1: 1st spectrum = 0th row etc.
        
        :param data: Spectral dataset to extract spectra from.
        :type data: np.array
        :param wavenumbers: Wavenumber array.
        :type wavenumbers: np.array
        :param spectrum_loc: Spectrum location.
        :type spectrum_loc: int or [int, int].

        :returns: numpy.array
        
        """
        #SPECTRUM_LOC MUST BE IN THE FORM OF AN INT or 1x2 ARRAY
        #Assumes spectra count starts at 1, not 0
        if type(spectrum_loc) == int:
            if data.ndim == 2:
                temp_data = data[(spectrum_loc-1),:]
            elif data.ndim == 3:
                dims, data = self.reshaper_3D(data)
                temp_data = data[(spectrum_loc-1),:]
            
        elif type(spectrum_loc) == list:
            if data.ndim == 2:
                dim1 = wavenumbers.shape[0]
                dim2 = self.ypixels
                dim3 = self.xpixels
                data = self.reshaper_3D(data, [dim2, dim3, dim1])
                temp_data = data[(spectrum_loc[0]-1),(spectrum_loc[1]-1),:]
            elif data.ndim == 3:
                temp_data = data[(spectrum_loc[0]-1),(spectrum_loc[1]-1),:]
                
        return temp_data
    
    def spectra_at(self, data, spectra_loc):
        """Extracts the absorbance spectra at a list of specified locations, 
        either as a list of [y,x] positions, or as a list of nth positions. 
        Function assumes user starts array at 1: 1st spectrum = 0th row etc.
        
        :param data: Spectral dataset to extract spectra from.
        :type data: np.array
        :param spectrum_loc: Spectra location.
        :type spectrum_loc: list of [int, ..., int] or list of [[int,int],...,[int,int]].

        :returns: numpy.array
        
        """
        
        #if coordinates are in [ypixel*xpixel,1] or  [ypixel*xpixel,] format 
        if (spectra_loc.ndim == 2 and spectra_loc.shape[1] == 1) or spectra_loc.ndim == 1: 
            if  data.ndim == 2:
                return data[spectra_loc,:]
            
            elif data.ndim ==3:
                dims, data = self.reshaper_3D(data)
                return data[spectra_loc,:]
        
        
        #if coordinates are in [ypixel, xpixel] format
        if spectra_loc.ndim == 2 and spectra_loc.shape[1] == 2:
            if  data.ndim == 2:
                data = self.reshaper_3D(data, [self.ypixels, self.xpixels, data.shape[1]])
                return data[spectra_loc[:,0],spectra_loc[:,1],:]
            
            elif  data.ndim == 3:
                return data[spectra_loc[:,0],spectra_loc[:,1], :]

        
    def area_at(self, position, data, wavenumbers):
        """Extracts the absorbance values at a specified wavenumber position.
        
        :param position: Wavenumber location.
        :type spectrum_loc: int

        :returns: numpy.array
        
        """
        is_3D = False
        if data.ndim == 3:
            dims, data = self.reshaper_3D(data)
            is_3D = True
        
        step = np.abs(wavenumbers[0]-wavenumbers[1])
        areas = data[:,(np.where((wavenumbers < position+step/2)*
                                  (wavenumbers > position-step/2) )[0][0])]
        
        if is_3D == True:
            areas = self.reshaper_2D(areas, dims[0:1])
        else:
            return areas
    
    def arg_size_checker(self, upper, lower):
        """Internal function to check the order of inputs so as to ensure 
        correct order for functions, essentialy swaps two inputs around in case
        the user gets the order wrong.
        
        :param upper: Upper argument.
        :type upper: int
        :param lower: Lower argument.
        :type lower: int

        :returns: int, int
        
        """
        if lower > upper:
            change = upper
            upper = lower
            lower = change      
        return upper, lower
    
    def area_under(self, data, **kwargs):
        """Internal wrapper for scipy's integration function.
        

        :returns: numpy.array
        
        """
        
        return sci_int.simps(data, **kwargs)
    
    def area_between(self, upper, lower, data=False, wavenums=False, **kwargs):
        """Extracts the area under each spectral absorbance plot between two
        specified wavenumber positions. Area is approximated using linear 
        baseline subtraction. kwargs passed are for any additional scipy
        integration arguments.
        
        :param upper: Upper wavenumber value.
        :type upper: int
        :param lower: Lower wavenumber value.
        :type lower: int
        :param data: Dataset input, default = False.
        :type data: numpy.array
        :param wavenums: Wavenumber input, default = False.
        :type wavenums: numpy.array

        :returns: numpy.array
        
        """
        #Get the ABS area within a wavenumber 
        upper, lower = self.arg_size_checker(upper, lower)
        
        if type(data) == bool:
            data = self.data
        if type(wavenums) == bool:
            wavenums = self.wavenumbers
        
        step = np.abs(wavenums[0]-wavenums[1])
        
        lower_loc = (np.where((wavenums < lower+step/2)*
                                  (wavenums > lower-step/2))[0][0])
        upper_loc = (np.where((wavenums < upper+step/2)*
                                  (wavenums > upper-step/2))[0][0])
        
        is_3D = False
        if data.ndim == 3:
            dims, data = self.reshaper_3D(data)
            is_3D = True
        
        data = data[:,lower_loc:upper_loc]
        data = self.all_spec_min2zero(data)
        
        areas = self.area_under(data, **kwargs)
               
        if is_3D == True:
            areas = self.reshaper_2D(areas, dims)
        
        return areas

    def overwrite_data(self, new_data, new_wavenumbers = 0):
        """Overwrites the object's data and wavenumber variables with inputted
        datasets.
        
        :param new_data: New data to insert.
        :type new_data: numpy.array
        :param new_wavenumbers: Optional new wavenumber data to insert, default = 0
        :type new_wavenumbers: numpy.array
        
        """
        self.data = new_data
        if type(new_wavenumbers) != int:
            self.wavenumbers = new_wavenumbers    
            
    def apply_mask(self, data, Mask):
        """Applies a logic mask to the data and extracts relevant spectra.
        
        :param data: Data to apply mask to.
        :type data: numpy.array
        :param Mask: Mask to apply.
        :type Mask: bool
        
        :returns: numpy.array
        
        """
        #Mask will be of type Bool  either in a nbym array or 1by(n*m) where
        #n and m are x and y pixels of the tile or mosaic           
        #self.data = self.data * Mask.astype(int)
        #n and m are x and y pixels of the tile or mosaic      
        if Mask.ndim ==2:
            dims, Mask = self.reshaper_2D(Mask)
            
        return data[Mask,:]
        
    
    def random_sample(self, Data, labels, size=0, seed=None, **kwargs):
        """Randomly assigns data samples and labels to training and test sets
        using scikitlearn's train_test_split function. All additional arguments
        are passable.
        
        :param Data: Spectral absorbance dataset. Of size [n_rows, m_features]
        :type Data: numpy.array
        :param labels: Label array of dataset. Of size [n_rows, 0]
        :type labels: numpy.array
        :param size: Number of samples to place in train set. Can either be a fraction in [0, 1] range, or an integer. Default = 0
        :type size: flaot32 | int
        :param seed: Optional seed argument for reproducability. Default = None
        :type seed: int
        :param kwargs: Additional train_test_split input arguments.
        :type kwargs: multiple
        
        :returns: X_train, X_test, y_test (numpy.array)
        
        """
        if type(seed) == int:
            np.random.seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(Data, labels, 
            random_state=seed, **kwargs)
        
        return X_train, X_test, y_train, y_test
    
    def leastsqr_correction(self, Orig_Data, Contam_spec, upper=3000, lower=2700):
        """Conducts Least Squares Regression subtraction to correct for a 
        contamination in the spectra, in example parrafin wax or wator vapor.
        Function allows for user specified wavenumber range for weight 
        calculations.
        
        :param Orig_Data: Original absorobance data needing correction.
        :type Orig_Data: numpy.array
        :param Contam_spec: A spectrum of contaminant.
        :type Contam_spec: numpy.array
        :param upper: Window upper wavenumber. Default = 3000.
        :type upper: int
        :param lower: Window lower wavenumber. Default = 2700.
        :type lower: int
        
        :returns: numpy.array
        
        """
        upper, lower = self.arg_size_checker(upper, lower)
        
        contam_data, _ = self.keep_range(upper, lower, Contam_spec, self.wavenumbers)
        data_snipped, _ = self.keep_range(upper, lower, Orig_Data, self.wavenumbers)
        
        x, resid, rank, s = np.linalg.lstsq(contam_data.reshape(len(contam_data[0,:]),1),data_snipped.reshape(len(data_snipped[0,:]),len(data_snipped[:,0])), rcond=None)

        subtract = np.dot(x.T, Contam_spec.reshape(1,-1))
        
        return Orig_Data - subtract
    
    def baseline_als(self, Data, w=0, lam=100, p=0.05, itermax=1):
        """Applies an Asymmetric Least Squares Smoothing baseline correction
        as outlined in Eiler and Boelens' 2005 paper:
        Eilers, P.H. and Boelens, H.F., 2005. Baseline correction with 
        asymmetric least squares smoothing. Leiden University Medical Centre 
        Report, 1(1), p.5.
        
        Function returns fitted baseline.
        
        :param Data: Absorbance spectrum to be baseline corrected.
        :type Data: numpy.array
        :param lam: Smoothness parameter for the baseline fit. Deafult = 10e1.
        :type lam: int
        :param p: Asymmetry parameter. Generally ranges from 0.001 - 0.1. Default = 0.005
        :type p: double
        :param itermax: Maximum iterations for fit.
        :type itermax: int
        
        :returns: z, w
        
        """
        
        L = Data.size
        D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
        if type(w) == int :
            w = np.ones(L)
        for i in np.arange(0,itermax):
              W = scipy.sparse.spdiags(w, 0, L, L)
              Z = W + lam * D.dot(D.transpose())
              z = scipy.sparse.linalg.spsolve(Z, w*Data)
              w = p * (Data > z) + (1-p) * (Data < z)
        return z, w


    def baseline_adaptive_PLS(self, x, lam=100, p=0.05, itermax=15):
        """Adaptive iteratively reweighted penalized least squares for baseline 
        fitting. This has key features taken from Zhimin Zhang's airPLS github
        code found at: https://github.com/zmzhang/airPLS
        The function has been adapted to work with the asymmetric least squares
        smoothing function as opposed to Whittaker Smoothing.
        Function returns fitted baseline to be subtracted from spectral 
        dataset.
        
        :param x: Spectrum of absorbance.
        :type x: numpy.array
        :param lam: Smoothness parameter for the baseline fit. Default = 10e1.
        :type lam: int
        :param p: Asymmetry parameter. Generally ranges from 0.001 - 0.1. Default = 0.005
        :type p: double
        :param itermax: Maximum iterations for fit.
        :type itermax: int
        
        :returns: fitted baseline z
        
        """
        
        m=x.shape[0]
        w=np.ones(m)
        
        for i in np.arange(0,itermax):
            z, w = self.baseline_als(x,w,lam,p)
            d = x-z
            dssn=np.abs(d[d<0].sum())
            if(dssn<0.001*(abs(x).sum()) or i==itermax):
                if i == itermax: 
                    print('WARING max iteration reached!')   
            
            w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
            w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
            w[0]=np.exp(i*(d[d<0]).max()/dssn) 
            w[-1]=w[0]
    
        return z

    def baseline_correct(self, Data, lam=100, p=0.05, itermax=15, mean=False, wavs=0):
        """Conducts baseline correcting for all spectra within the dataset.
        Can be set to baseline fit on the mean of all spectra to vastly decrease
        computational load.
        
        :param Data: Spectral Dataset to be baseline fitted.
        :type Data: numpy.array
        :param lam: Smoothness parameter for the baseline fit. Default = 10e1.
        :type lam: int
        :param p: Asymmetry parameter. Generally ranges from 0.001 - 0.1. Default = 0.005
        :type p: double
        :param itermax: Maximum iterations for fit.
        :type itermax: int
        :param mean: Decision whether to mean all spectra or us each spectrum
        :type mean: Boolean
        
        :returns: Correcred dataset, fitted baseline z. numpy.array
        
        """
        
        if Data.ndim == 2:
            D_corrected = np.zeros((Data.shape[0], Data.shape[1]))
        elif Data.ndim == 1:
            D_corrected = np.zeros(Data.size)
            Data = Data.reshape(1,-1)
        if mean == False:
            for i in np.arange(0, Data.shape[0]):
                d = Data[i,:]
                z = self.baseline_adaptive_PLS(d, lam, p, itermax)
                D_corrected[i,:] = d-z
        elif mean == True:
            #Going to scale the subtraction to match the AmideI peak
            d = np.mean(Data, axis=0)
            z = self.baseline_adaptive_PLS(d, lam, p, itermax)
            if type(wavs) != int:
                mean_peak = self.area_between(1700, 1600, data=d.reshape(1,-1), 
                                              wavenums=wavs)
                peak_scales = self.area_between(1700, 1600, data=Data, 
                                                wavenums=wavs)
                peak_scales = peak_scales/mean_peak
                z = np.dot(peak_scales.reshape(1,-1).T,z.reshape(1,-1))
                
            D_corrected= Data-z
            
        return D_corrected, z

    def baseline_parameter_test(self, data, lam=100, p=0.05):
        """Plots overlapping plots of the original mean spectrum, fitted 
        baseline, and approximated corrected spectrum
        
        :param Data: Spectral Dataset to be baseline fitted, either meaned or not.
        :type Data: numpy.array
        :param lam: Smoothness parameter for the baseline fit. Default = 10e1.
        :type lam: int
        :param p: Asymmetry parameter. Generally ranges from 0.001 - 0.1. Default = 0.005
        :type p: double
        
        :returns: Matplotlib.pyplot, lam(double), p(double)
        
        """
        
        if data.ndim ==2:
             data = np.mean(data, axis=0)
             
        _, z = self.baseline_correct(data, lam, p, mean=True)
        
        t = np.arange(0, data.size)
        
        plt.figure()
        plt.plot(t, data, label = "Original spectrum")
        plt.plot(t, z, label = "Fitted baseline")
        plt.plot(t, data-z, label = "Final spectrum")
        plt.ylabel('y - absorbance')
        plt.title('Approximate baseline subtraction')
        plt.legend()
        plt.show()
        
        return lam, p
    
    def ME_EMSC_Correction_Solver(self, Raw_Spectra, Reference, Contaminant, lamb=1000, p_=0.01):
        """
        ME_EMSC_Correction_Solver is built to implement corrections to solve for 
        contaminants in spectra, in example the removal of paraffin wax from 
        a spectral dataset.
        Base solver code adapted from Johanne Solheim's Matlab function code
        found at: https://github.com/BioSpecNorway/ME-EMSC/blob/master/computing/EMSC/ME_EMSCsolver.m
        
        Implement corrections to solve for contaminants in spectra, 
        in example the removal of paraffin wax from a spectral dataset.
        
        Baseline P is calculated using Assymetric Least Squares Regression.
        
        EMSC model is build of a matrix contaning elements of the model as
        column vectors:
            0 - consistent baseline
            1 - reference spectrum (pure biological reference)
            2:end - PCA loadings from PCA model built on Contaminants.
        
        :param Raw_Spectra: Spectral Dataset to be corrected
        :type Raw_Spectra: numpy.array
        :param Reference: Reference Spectrum for fitting, must be independent and without contaminant
        :type Reference: numpy.array
        :param Contaminant: Contaminant Spectral dataset, more samples the better.
        :type Contaminant: numpy.array
        :param lam: Smoothness parameter for the baseline fit. Default = 1000.
        :type lam: int
        :param p: Asymmetry parameter. Default = 0.01
        :type p: double
        
        
        :returns: Corrected(np.array), Residuals(np.array)
        
        """
        
        ## Step 1 - build paraffin model based on pca ~ 98% cum explained var
        temp_pca = pyir_pca.PyIR_PCA()
        temp_pca.fit_PCA(Contaminant, n_comp=50)
        # n_comps selected for that which obtains 97.5% cum explained var
        n_comp = np.where(np.cumsum(temp_pca.pca_module.explained_variance_ratio_)
                          > 0.975)[0][0]
        model = temp_pca.pca_loadings[0:n_comp]
        
        ## Step 2 - calculate baseline using ALS subtraction
        _, P = self.baseline_correct(Data=Raw_Spectra, lam=lamb, p=p_, mean=True)
        
        ## Step 3 - build EMSC MODEL  as outlined in docs 
        EMSCModel = np.column_stack((P, Reference, np.moveaxis(model,0,-1)))
        
        ## Step4 - solve for unknown parameters using OLS
        params, resid, rank, s = np.linalg.lstsq(EMSCModel, 
                                np.moveaxis(Raw_Spectra,0,-1), rcond=None)

        Corrected = (Raw_Spectra - np.moveaxis(np.dot(EMSCModel,params),0,-1)+ 
        np.reshape(params[1,:],(params.shape[1],1)) * np.reshape(EMSCModel[:,1], 
                                                    (1,EMSCModel.shape[0])))

        return Corrected, Raw_Spectra-Corrected  
    
    def resid_sum_sq(self, data, prin_comps=50):
        """Residual sum of squares statistic calculation tool taken from 
        Chemometrics for Pattern Recognition by Richard Brereton. Purpose
        is for cross validation to confirm how many PC's explain the chemical
        information conatined in a dataset.
        
        :param data: Dataset of spectra (must be in row format)
        :type data: np.array
        :param prin_comps: Numper of principal components to get RSS stats for.
        :type prin_comps: int. Default = 50.

        
        :returns: rss_values (np.array), rss_predictions (np.array)

        """
        
        rows = data.shape[0]
        cols = data.shape[1]
        
        rss_predictions = np.empty((rows,cols, prin_comps))
        rss_values = np.empty((prin_comps,1))
        
        #mean centering occurs in the pca toolset but need the mean ourselves
        data_mean = np.mean(data, axis=0)
        data_centered = (data - data_mean)
        
        temp_pca_object = pyir_pca.PyIR_PCA()
        temp_pca_object.fit_PCA(data_centered, n_comp = prin_comps)
        loadings = temp_pca_object.pca_loadings
        scores = temp_pca_object.pca_scores
        
        predicted_scores = np.dot(data_centered, loadings.T)
        
        for x in np.arange(1, prin_comps+1):
            #loadings need to be transposed so dimensions match
            predicted_data = np.dot(predicted_scores[:,0:x],
                      loadings[0:x,:])
            #re-add mean
            predicted_data = predicted_data + data_mean
            rss_predictions[:,:, x-1] = predicted_data
        
        for x in np.arange(0, prin_comps):
            rss_value = np.sum(np.sum((rss_predictions[:,:,x] - data)* 
                                      (rss_predictions[:,:,x] - data)))
            rss_values[x] = rss_value
        
        return rss_values, rss_predictions
    
    def PRESS(self, data, prin_comps=50):
        """Predicted Residual Sum of Squares (PRESS) statistic calculation 
        tool taken from Chemometrics for Pattern Recognition by Richard 
        Brereton. Purpose is for cross validation to confirm how many PC's 
        explain the chemical information conatined in a dataset.
        
        :param data: Dataset of spectra (must be in row format)
        :type data: np.array
        :param prin_comps: Numper of principal components to get RSS stats for.
        :type prin_comps: int. Default = 50.

        
        :returns: rss_values (np.array), rss_predictions (np.array)

        """
        rows = data.shape[0]
        cols = data.shape[1]
        
        press_predictions = np.empty((rows,cols, prin_comps))
        press_values = np.empty((prin_comps,1))
        non_test = np.arange(0,rows)

        for x in np.arange(0, rows):
            #exclude the test spectrum from the training set
            test_set = data[x,:]
            training_set = data
            training_set = training_set[non_test!=x ,:]
            
            #mean center
            training_mean = np.mean(training_set, axis=0)
            training_set = training_set - training_mean
            #remove mean from test
            test_set = test_set - training_mean
            
            temp_pca_object =  pyir_pca.PyIR_PCA()
            temp_pca_object.fit_PCA(training_set, n_comp = prin_comps)
            loadings = temp_pca_object.pca_loadings
            scores = temp_pca_object.pca_scores
            
            predicted_test_scores = np.dot(test_set, loadings.T).reshape(1,-1)
            
            for pc in np.arange(1, prin_comps+1):
                #loadings need to be transposed so dimensions match
                predicted_data = np.dot(predicted_test_scores[:,0:pc],
                          loadings[0:pc,:])
                #re-add mean
                predicted_test_data = predicted_data + training_mean
                press_predictions[x,:, pc-1] = predicted_test_data

        for x in np.arange(0, prin_comps):
            press_value = np.sum(np.sum((press_predictions[:,:,x] - data)* 
                                      (press_predictions[:,:,x] - data)))
            press_values[x] = press_value
    
        return press_values, press_predictions
    
    def PRESS_RSS_test(self, data, prin_comps=50, plot=True):
        """PRESS/RSS ratio to determine the number of principal components to
        use in this data when smoothing to eliminate noise. This approach
        is taken from Chemometrics for Pattern Recognition by Richard Brereton.
        The output value is the first principal component value where the
        PRESS/RSS ratio is greater than 1.
        
        :param data: Dataset of spectra (must be in row format)
        :type data: np.array
        :param prin_comps: Numper of principal components to get RSS stats for.
        :type prin_comps: int. Default = 50.
        :param plot: Toggle for plotting PRESS/RSS ratio curve
        :type plt: int. Bool.
        
        :returns: pc (int)

        """
        
        press_values, press_predictions = self.PRESS(data, prin_comps)
        rss_values, rss_predictions = self.resid_sum_sq(data, prin_comps)
        
        press_rss_ratio = press_values[1:]/rss_values[0:rss_values.shape[0]-1]
        pcs = np.where(press_rss_ratio >1)[0][0]
        
        if plot==True:
            plt.figure()
            plt.plot(np.arange(1,rss_values.shape[0]), press_rss_ratio)
            plt.xlabel("principal components")
            plt.ylabel("PRESS/RSS ratio")
        
        return pcs
    
    def DRMAD_test(self, data):
        """Determination of Rank by Median Absolute Deviation (DRMAD)
        Taken from paper by E.R. Malinokswi
        "Determination of rank by median absolute deviation (DRMAD): a simple
        method for determining the number of principal factors responsible for 
        a data matrix" Journal of Chemometrics 23 (2009) 1-6
        http://dx.doi.org/10.1002/cem.1182
        
        :param data: Dataset of spectra (must be in row format)
        :type data: np.array

        
        :returns: Rank (int)

        """
        
        def MAD_TEST(matrix):
            n_p = matrix.shape[0]
            medianx = np.median(matrix)
            dx = matrix - (medianx * np.ones((1,n_p)))

            MADx = np.median(np.abs(dx))
            if MADx ==0:
                MADx = 0.00000001
            #abs(1,1) means UPPER MOST input value!!! Therefore just matrix[0]
            test = (np.abs(matrix[0] - medianx))/MADx
            out = 1  
            if test > 5:
                out = 0
            
            return out
        
        rows = data.shape[0]
        cols = data.shape[1]
        sm = cols
        if rows < cols:
            sm = rows 
        
        #Data needs to be mean centered I think
        data_mean = np.mean(data, axis=0)
        data = data - data_mean
        
        u, s, vh = np.linalg.svd(data, full_matrices=True)

        n = np.arange(0,sm)
        ev = np.zeros((sm))
        sev = np.zeros((sm))
        rsd = np.zeros((sm))
        for j in np.arange(0,sm):
            #S IS THE DIAGANOL ELEMENTS, THEREFORE S[J,J] BECOMES JUST S[J]
            ev[j] = s[j]*s[j]
        for k in np.arange(0,sm):
            sev[k] = np.sum(ev[k:sm])   
        for jj in np.arange(0, sm):
            rsd[jj] = np.sqrt(sev[jj]/((rows-jj+1)*(cols-jj+1)))
         
        test = np.zeros((sm))
        for x in np.arange(0, sm):
            test[x] = MAD_TEST(rsd[x:sm])      
            
        results = np.empty((sm, 4))
        results[:,0] = n
        results[:,1] = ev
        results[:,2] = rsd
        results[:,3] = test
        
        return results
    
    def stitch_tiles_2(self, tile_1, tile_2, dims=[128, 128], direction = "hor"):
        """Stitches two individual FTIR tiles in to a single image, will always 
        stitch horzintally by default but can stitch vertically. Images must be the
        same dimensions.
        
        :param tile_1: First FTIR tile to stitch. Requires n*x*v array.
        :type tile_1: np.array.
        :param tile_2: Second FTIR tile to stitch. Requires n*x*v array.
        :type tile_2: np.array.
        :param direction: Direction of concatenation, can be horizontal or vertical. Default = "hor".
        :type direction: string.
        
        :returns: numpy array of bool.
        
        """ 
        
        is_3D = False
        
        if tile_1.ndim ==2:
            tile_1 = self.reshaper_3D(tile_1,reshapedim = dims)
            is_3D = True
            
        if tile_2.ndim ==2:
            tile_2 = self.reshaper_3D(tile_2, reshapedim = dims)
            is_3D = True
        
        if tile_1.shape[0] != tile_2.shape[0] or tile_1.shape[1] != tile_2.shape[1]:
            print("Dimensions of tiles do not agree")
            return
        
        if direction == "hor" or direction == "horizontal":
            ypixels = dims[0]
            xpixels = dims[1]*2
            
            stitched_data = np.concatenate((tile_1, tile_2), axis=1)
            
        elif direction == "ver" or direction == "vertical":
            ypixels = dims[0]*2
            xpixels = dims[1]
            
            stitched_data = np.concatenate((tile_1, tile_2), axis=0)


        stitched_image = PyIR_SpectralCollection()
        if is_3D == True:
            _, stitched_image.data = self.reshaper_3D(stitched_data)
        else:
            stitched_image.data = stitched_data
        stitched_image.xpixels = xpixels
        stitched_image.ypixels = ypixels
        stitched_image.update_sums()
        
        return stitched_image


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
        if ypixels ==0:
            ypixels = self.ypixels
        if xpixels ==0:
            xpixels = self.xpixels
            
        return pyir_image.PyIR_Image.disp_image(**locals())