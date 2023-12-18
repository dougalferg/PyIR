"""
    .. module:: pyir_pccva
       :platform: Windows
       :synopsis: principal component led canonical variate analysis,

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import numpy as np
import sklearn

class PyIR_PCCVA:
    """PyIR_PCCVA is a class to be used for the principal component led 
    canonical variate analysis based off of "A modification of canonical 
    variates analysis to handlehighly collinear multivariate data" 
    https://doi.org/10.1002/cem.1017
        
    It uses super() inheritance to take all functions from PY_IR modules.
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def undersample(self, spectra, class_numbers, labels):
        #count the number of datapoints per class
        class_counts = np.empty((class_numbers), dtype = int)
        for group in np.arange(0, class_numbers):
            class_counts[group] = np.sum(labels==group)
        #undersample amount is that of the smallest group
        to_sample = int(np.min(class_counts))
        #generate sampling mask for spectra
        sampling_mask = np.zeros(int((np.sum(class_counts))), dtype = bool)
        count=0
        for group in np.arange(0, class_numbers):
            group_sample = sampling_mask[count:count+class_counts[group]]
            random_choice = np.random.choice(class_counts[group], to_sample, 
                                             replace=False)
            group_sample[random_choice] = True
            sampling_mask[count:count+class_counts[group]] = group_sample
            count = count+class_counts[group]
            
        #undersample spectra and rebuild labels ?
        spectra = spectra[sampling_mask, :]
        count = 0 
        labels = np.empty((spectra.shape[0]), dtype=int)
        for i in np.arange(0, class_numbers):
            labels[count:(count+
                             to_sample)] = i
            count = (count+ to_sample)
        
        return spectra, labels
    
    
    def fit_PCCVA(self, spectra, labels=0, pcs=0, sampling="under"):
        """Fits a PCCVA model to the spectral data, requires labels for the 
        dataset. The CVA model is built by default on the scores of a PCA model
        that explain 95% variance of the spectral dataset.
        
        :param spectra: Data to fit PCCVA model to, expects a nxv spectral dataset.
        :type spectra: array of flat64
        :param labels: Array of leabels for the dataset, expects an array of int. Default = 0
        :type labels: array of int32 or equivalent.
        :param pcs: Number of PCA principal components to calculate. Default is pcs = 95% variance.
        :type pcs: int
        :param sampling: Sampling argument. Default = "under"
        :type sampling: string
        
        
        :returns: PCCVA movdel.
        
        """
        
        #check labels were provided
        if type(labels) == int :
            print("No class labels, these are required for the function to work")
            return 
        
        #calculate the number of unique class labels
        number_of_classes = len(np.arange(np.min(labels), np.max(labels)+1))
        
        #undersample to create even sample sizes
        if sampling == "under":
           spectra, labels = self.undersample(spectra, number_of_classes, labels)
            
        #fit pca model
        pca_module = sklearn.decomposition.PCA().fit(spectra)
        pca_mean = np.mean(spectra, axis=0)
        #determine valid PC's
        if pcs==0:
            pcs = np.where(np.cumsum(pca_module.explained_variance_ratio_) > 0.95)[0][0]
        
        #get loadings and scores for the pca at the specified pcs
        pca_loadings = pca_module.components_[0:pcs,:]
        pca_scores = np.dot(spectra,pca_loadings.T)
        
        #determine number of valid canonical variates
        num_cvs = number_of_classes-1
        if pcs < num_cvs:
            print("Not eonugh valid principal components to discriminate")
            return
        
        #perform canonical variates analysis
        cveigenvalues, cveigenvectors, cvpercent_explained_variation = self.fit_CVA(pca_scores,
                                            labels, sampling)
        
        cvloadings = np.dot(pca_loadings.T, cveigenvectors)
        temp_scores = np.dot(cveigenvectors,np.diag(cveigenvalues))
        cvscores = np.dot(pca_scores, temp_scores)
    
        return {"CVAloadings": cvloadings,
                "CVAscores": cvscores,
                "eigenvalues": cveigenvalues,
                "eigenvectors": cveigenvectors,
                "explained variance": cvpercent_explained_variation,
                "n_pcs": pcs,
                "pca_model": pca_module,
                "pca_mean": pca_mean}
      
        
        
    def fit_CVA(self, pca_scores, labels=0, sampling="under"):
        """Fits a CVA model to the data, requires labels for the 
        dataset. The CVA model is built by default on the scores of a PCA model.
        The function eigenvalue eigenvector pairing that creates the most 
        separation between groups.
        
        :param pca_scores: PCA scores input. Does not have to be pca_scores.
        :type pca_scores: array of flat64
        :param labels: Array of leabels for the dataset, expects an array of int. Default = 0
        :type labels: array of int32 or equivalent.
        :param sampling: Sampling argument. Default = "under"
        :type sampling: string
        
        
        :returns: eigenvalues, eigenvectors, explained_variance
        
        """
        #check labels were provided
        if type(labels) == int:
            print("No class labels, these are required for the function to work")
            return 
        
        #calculate the number of unique class labels
        number_of_classes = len(np.arange(np.min(labels), np.max(labels)+1))
        
        #number of canonical variates
        num_cvs = number_of_classes-1
        if num_cvs <=0:
            print("Not enough classes for canonical variate analysis.")
            return
        
        #get number of scores and wavenumber points 
        numscores, numpts = pca_scores.shape
        
        #calculate within-group-covariance
        within_group_covariances = np.zeros((numpts, numpts, number_of_classes))
        for group in np.arange(0, number_of_classes):
            group_data = pca_scores[labels==group, :]
            group_mean = np.mean(group_data, axis=0).reshape(1,-1)
            mean_centered_group = group_data - group_mean
            covariance = np.dot(mean_centered_group.T, mean_centered_group) 
            within_group_covariances[:,:, group] = covariance
        
        within_group_covariance = np.sum(within_group_covariances, axis=2)/(
            numscores-number_of_classes)
        
        #calculate between-group-covariance
        between_group_covariances = np.zeros((numpts, numpts, number_of_classes))
        data_mean = np.mean(pca_scores, axis=0)
        for group in np.arange(0, number_of_classes):
            group_data = pca_scores[labels==group, :]  
            group_mean = np.mean(group_data, axis=0).reshape(1,-1)
            
            mean_diff = group_mean - data_mean
            covariance = group_data.shape[0] * np.dot(mean_diff.T, mean_diff)
            
            between_group_covariances[:,:, group] = covariance
        
        between_group_covariance = np.sum(between_group_covariances, axis=2)/(
            number_of_classes-1)
        
        eigenvalues, eigenvectors  = np.linalg.eig(np.linalg.lstsq(within_group_covariance, 
                                                  between_group_covariance, rcond=None)[0])
    
        # Construct the output
        idx = eigenvalues.argsort()     # get index values that will sort eigenvalues in ascending order
        idx = np.flip(idx)              # flip to descending order
        eigenvalues = eigenvalues[idx]
        eigenvalues = np.real(eigenvalues[0:num_cvs])
    
        eigenvectors = eigenvectors.T
        eigenvectors = eigenvectors[idx, :]
        eigenvectors = np.real(eigenvectors[0:num_cvs, :])
        eigenvectors = eigenvectors.T
    
        explained_variation = eigenvalues / sum(eigenvalues)
    
        return eigenvalues, eigenvectors, explained_variation

