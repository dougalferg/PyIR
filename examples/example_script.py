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


tile_1 = pir.PyIR_SpectralCollection(r'\\nasr.man.ac.uk\epsrss$\snapped\replicated\gardner\Masters and UG Projects\2021-2022\Dilara and Amelia\PR484_C051\PR484_C051_F7\pr484_c051_f7.dmt')
tile_1.load_data()
tile_1.update_sums()


temp_pca = pir_pca.PyIR_PCA()
temp_pca.fit_PCA(tile_1.data, n_comp=60)

tile_1.data = temp_pca.noise_reduction(tile_1.data, n_comps=60)

k_means = tile_1.K_means_cluster(tile_1.area_between(1750,1500, 
                        tile_1.data, tile_1.wavenumbers).reshape(-1,1), clusters=2)
k_means_pred = k_means.labels_


mask = tile_1.disp_image(k_means_pred)

Group_0_mean = np.reshape(np.mean(tile_1.apply_mask(tile_1.data, k_means_pred == 0), axis=0),(1,tile_1.wavenumbers.shape[0]))
Group_1_mean = np.reshape(np.mean(tile_1.apply_mask(tile_1.data, k_means_pred == 1), axis=0),(1,tile_1.wavenumbers.shape[0]))

areas = tile_1.area_between(1750, 1500, 
                          np.concatenate((Group_0_mean,Group_1_mean)), 
                          tile_1.wavenumbers)

wax_mask = (k_means_pred == np.where(areas == np.min(areas))[0][0])
tissue_mask = (k_means_pred == np.where(areas == np.max(areas))[0][0])

tissue_section = tile_1.tissue_excluder(tissue_mask)  

tile_1.disp_image(tissue_section)





