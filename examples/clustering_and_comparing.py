import sys
import os

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\PyIR\src'
sys.path.append(full_path)


import pyir_spectralcollection as pir
import pyir_image as pir_im
import pyir_mask as pir_mask
import pyir_pca as pir_pca
import numpy as np


## Loading in the spectral dataset
tile_1 = pir.PyIR_SpectralCollection(r'\\nasr.man.ac.uk\epsrss$\snapped\replicated\gardner\Masters and UG Projects\2021-2022\Dilara and Amelia\PR484_C051\PR484_C051_F7\pr484_c051_f7.dmt')
tile_1.load_data()
tile_1.update_sums()


#replicate loading steps to obtain tissue spectra
## Identification of tissue spectra by clustering the Amide I and II peaks
k_means = tile_1.K_means_cluster(tile_1.area_between(1750,1500, 
                        tile_1.data, tile_1.wavenumbers).reshape(-1,1), clusters=2)
k_means_pred = k_means.labels_

# Create separate wax and tissue masks from the Amide I and II area
# under curves (AUC)
Group_0_mean = np.reshape(np.mean(tile_1.apply_mask(tile_1.data, k_means_pred == 0), axis=0),(1,tile_1.wavenumbers.shape[0]))
Group_1_mean = np.reshape(np.mean(tile_1.apply_mask(tile_1.data, k_means_pred == 1), axis=0),(1,tile_1.wavenumbers.shape[0]))

areas = tile_1.area_between(1750, 1500, 
                          np.concatenate((Group_0_mean,Group_1_mean)), 
                          tile_1.wavenumbers)

wax_mask = (k_means_pred == np.where(areas == np.min(areas))[0][0])
tissue_mask = (k_means_pred == np.where(areas == np.max(areas))[0][0])

# Additional tissue fragments can be removed using the tissue_exlcluder
# function
tissue_section = tile_1.tissue_excluder(tissue_mask)  

#Extract the tissue data, pre-process, calculate derivatives, and plot.
tissue_spec = tile_1.apply_mask(tile_1.data, tissue_section)
tissue_spec, tissue_spec_wavenumbers = tile_1.remove_wax(tissue_spec, tile_1.wavenumbers)
tissue_spec, tissue_spec_wavenumbers = tile_1.keep_range(1800, 1000, tissue_spec, tissue_spec_wavenumbers)
tissue_spec = tile_1.all_spec_min2zero(tissue_spec)
tissue_spec = tile_1.vector_norm(tissue_spec)

#cluster groups within the data
k_means = tile_1.K_means_cluster(tissue_spec, clusters=5)
k_means_pred = k_means.labels_

k_means_image = tile_1.cluster_rebuild(k_means_pred, tissue_section)

#plot original image, tissue mask, and clusters map
tile_1.disp_image(tile_1.totalimage, title = "raw image")
tile_1.disp_image(tissue_section, title = "tissue mask")
tile_1.disp_image(k_means_image, title = "clusters") #, colour_bar = True)

## extract and plot spectral groups
spectra_ordered = np.empty((1, tissue_spec_wavenumbers.shape[0]))
for i in np.arange(1, np.max(k_means_pred)+1):
    spectra_ordered = np.concatenate((spectra_ordered, 
                        tile_1.apply_mask(tissue_spec, k_means_pred==i)))
spectra_ordered = spectra_ordered[1:,:]

#create label group for plotting
labels = tile_1.make_class_labels(['Cluster 1', 'Cluster 2', 'Cluster 3',
                                   'Cluster 4', 'Cluster 5'],
                [np.sum(k_means_pred==1),np.sum(k_means_pred==2),
                 np.sum(k_means_pred==3),np.sum(k_means_pred==4),
                 np.sum(k_means_pred==5)])

#Plot groups of spectra
tile_1.disp_conf_int(spectra_ordered, tissue_spec_wavenumbers, labels)