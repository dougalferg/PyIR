"""
    .. module:: pyir_pca
       :platform: Windows
       :synopsis: PCA module for all things principal component analysis with FT-IR Spectral data

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pyir_spectralcollection


class PyIR_PCA:
    """PyIR_PCA is a class to be used mainly for the 
    usage of PCA tools on the spectral data. This class
    uses super() inheritance to take all functions from Doug_Toolbox().
    Most functions are wrappers of the sklearn.decomposition module 
    paired with other toolbox functions.

    """
    def __init__(self, *args, **kwargs):
        
        self.ubermask = []
        self.pca_module = []
        self.pca_scores = []
        self.pca_loadings = []
        super().__init__(*args, **kwargs)

    def get_vals(self, toolbox_object):
        """ Inherits required data from a specfied toolbox_object. Wavenumbers
        and image dimensions are inherited.

        """
        self.wavenumbers = toolbox_object.wavenumbers
        self.xpixels = toolbox_object.xpixels
        self.ypixels = toolbox_object.ypixels 
        
    def get_labels(self, toolbox_object):
        """ Inherits label objects from a specfied toolbox_object.

        """
        self.class_labels = toolbox_object.class_labels
        self.class_label_names = toolbox_object.class_label_names
        
    def set_labels(self, Labels):
        """ Sets the pca labels based upon a labels object
        """
        
        self.class_label_names  = Labels[0]
        self.class_labels= Labels[1]
    
    def fit_PCA(self, data, n_comp = 50):
        """Fits a PCA model to the data, using n_components.
        
        :param data: Data to fit PCA to.
        :type data: numpy.array
        :param n_comp: Numper of principal components to fit to. Default = 50
        :type n_comp: int
        
        
        :returns: object.pca_module (.PCA()), object.pca_loadings (numpy.array)
        
        """
        #Checks to see if input is 3D.
        if data.ndim == 3:
            dims, data = pyir_spectralcollection.reshaper_3D(data)
        self.pca_module = sklearn.decomposition.PCA(n_components=n_comp).fit(data)
        self.pca_loadings = self.pca_module.components_
        self.pca_scores = np.dot(data, 
                                 self.pca_module.components_[0:n_comp,:].T)
        
    def plot_pca_image(self, prin_comp = 1, norm = True):
        """Plots an image of specific principal component scores.
        
        :param prin_comp: Which principal component to use in plot. Default = 1
        :type prin_comp: int
        :param norm: Additional argument to normalise the loadings. Default = True
        :type norm: Bool
        
        
        :returns: matplotlib.pyplot plot
        
        """
        data = np.array(self.pca_module.components_[prin_comp-1,:])
        if norm == True:
            data = 2*((data-np.min(data))/(np.max(data)-np.min(data))) - 1
        
        data = data.reshape(self.ypixels, self.xpixels)
        
        title = ('Score on Principal Component ' + str(prin_comp))
        self.disp_image(data, title, colour_bar = True)
        
    def plot_scores(self, data, prin_comp1, prin_comp2):
        """Plots principal component scores on a matplotlib.pyplot scatter plot.
        
        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param prin_comp1: Which principal component to use in plot x-axis.
        :type prin_comp1: int
        :param prin_comp2: Which principal component to use in plot y-axis.
        :type prin_comp2: int
        
        
        :returns: matplotlib.pyplot plot
        
        """
        plt.figure()
        colour_map = plt.cm.get_cmap('rainbow')
        norm = plt.Normalize(vmin= self.class_labels.min(), 
                             vmax = self.class_labels.max())
        self.class_labels = (self.class_labels.astype('float64')).ravel()
        
        x_all = np.dot(data, self.pca_module.components_[prin_comp1-1,:])
        y_all = np.dot(data, self.pca_module.components_[prin_comp2-1,:])
        
        x_std = np.std(x_all)
        y_std = np.std(y_all)
        
        #need it to loop through for each "group"
        temp_low = 0
        for i in range(0, len(self.class_label_names)):
            temp_high = temp_low + np.sum(self.class_labels == i)
            plt.scatter(x_all[temp_low:temp_high], y_all[temp_low:temp_high],
                    c = colour_map(norm(self.class_labels)[temp_low:temp_high]),
                    label = self.class_label_names[i][:])
            temp_low = temp_high
        
        plt.xlabel('PC ' + str(prin_comp1) + ' (' +
                   str(self.pca_module.explained_variance_ratio_
                       [prin_comp1-1]*100)+ ') % var')
        plt.ylabel('PC ' + str(prin_comp2) + ' (' +
                   str(self.pca_module.explained_variance_ratio_
                       [prin_comp2-1]*100)+ ') % var')
        plt.ylim(np.min(y_all) - y_std,
                 np.max(y_all) + y_std)
        plt.xlim(np.min(x_all) - x_std,
                 np.max(x_all) + x_std)
        
        plt.legend(loc = 'best')
        plt.show()
       
    def plot_loading(self, prin_comp, wavenums=0):
        """Plots principal component loading across the wavenumber range.
        
        :param prin_comp: Which principal component to use in plot y-axis.
        :type prin_comp: int     
        :param wavenums: Wavenumber array to plot loading against.
        :type wavenums: numpy.array     
        
        
        :returns: matplotlib.pyplot plot
        
        """
        
        temp=0
        if type(wavenums) == int:
            temp = 1
            wavenums = np.arange(0, self.pca_loadings[prin_comp-1,:].shape[0])
        
        plt.figure()
        plt.plot(np.ravel(wavenums), self.pca_loadings[prin_comp-1,:])
        plt.ylabel('weight')
        plt.xlabel('wavenumber cm-1')
        if temp == 1:
            plt.xlabel('array position')
        plt.title('PCA Loading for Principal Component ' + str(prin_comp)) 
        plt.show()

        
    
    def plot_cum_explained_var(self, max_prin_comps = 15):
        """Plots cumulative explained variance of the principal components.
        
        :param max_prin_comps: Maximum principal components. Default = 15.
        :type max_prin_comps: int      
        
        :returns: matplotlib.pyplot plot
        
        """
        plt.figure()
        plt.plot(np.cumsum(self.pca_module.explained_variance_ratio_[0:
                                                            max_prin_comps]))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        
    def dimension_reduce(self, data, n_comps=20):
        """Reduces dataset to user defined n_components.
        
        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param n_comps: Number of principal components. Default = 20.
        :type n_comps: int   
        
        :returns: numpy array
        
        """  
        if type(self.pca_module) == list:
            self.fit_PCA(data)
        
        #Reduce eigenvector matrix down to number of comps selected
        temp_components = self.pca_module.components_
        self.pca_module.components_ = self.pca_module.components_[0:n_comps,:]
        data = self.pca_module.transform(data)
        
        #return the original loadings just in case
        self.pca_module.components_ = temp_components
        
        return data
    
    def noise_reduction(self, data, n_comps=10):
        """Utilises PCA noise reduction technique.
        
        :param data: Original data to be transformed by each princ_comp.
        :type data: np.array
        :param n_comps: Number of principal components used in noise reduction. Default = 10.
        :type n_comps: int   
        
        :returns: numpy array
        
        """     
        return np.dot(self.pca_scores[:,0:n_comps],
                          self.pca_loadings[0:n_comps,:])
    
    def kernel_PCA(self, n_comps=None, kernel='linear', **kwargs):
        """Creates a KernelPCA object for non-linear dimensionality reduction using
        KernalPCA module. Once created, the KernelPCA functions can be then used.
        Refer to scikitlearn KernelPCA documentation for additional commands.
        
        :param n_comps: Number of principal components used.
        :type n_comps: int   
        :param kernel: Kernel to be used from 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
        :type kernel: string   
        :param **kwargs: Additional inputs of KernelPCA module.
        :type **kwargs: multiple.
        
        :returns: decomposition.KernalPCA object. Refer to scikitlearn docs.
        
        """
        
        return sklearn.decomposition.KernelPCA(n_components = n_comps, kernel = kernel, **kwargs)   
    
    def transform_dims(self, data, loadings, n_comp=0):
        """Transforms the dataset to a specified number of components from
        an external loadings array.
        
        :param data: Original data to be dimensionally transformed.
        :type data: np.array
        :param loadings: Loadings array from a Doug_PCA object.
        :type loadings: np.array 
        :param n_comp: User defined number of components, default=0.
        :type n_comp: int
        
        
        :returns: numpy array
        
        """  
        
        if n_comp !=0: 
            #Reduce eigenvector matrix down to number of comps selected
            loadings = loadings[0:n_comp]
        
        #Project data into now dimensional space
        data = np.dot(data, loadings.T)
        
        return data