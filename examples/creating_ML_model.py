#### Tutorial on how to implement any machine learning application from data
#obtained and handled using PyIR


## Model will be a simple RF classifier for tissue and wax
import sys
import os

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\PyIR\src'
sys.path.append(full_path)


import pyir_spectralcollection as pir
import numpy as np

import sklearn


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


#Pre-process
tile_1.data, tile_1.wavenumbers = tile_1.keep_range(3500, 1000, tile_1.data, tile_1.wavenumbers)
tile_1.data = tile_1.all_spec_min2zero(tile_1.data)
tile_1.data = tile_1.vector_norm(tile_1.data)

#Randomly subsample out "tissue" and "wax" groups using tissue and wax
#masks

#create combined image of labels
label_mask = np.zeros((wax_mask.shape[0])).astype(int)
label_mask[tissue_mask] = 1
label_mask[wax_mask] = 2
tile_1.disp_image(label_mask)

#Create model
test_x, train_x, test_y, train_y = tile_1.random_sample(
                            tile_1.data, label_mask, seed=123, test_size=2000)

RF_classifier = sklearn.ensemble.RandomForestClassifier().fit(train_x, train_y)

#Predict
RF_predict = RF_classifier.predict(test_x)

##correct wax
np.sum(RF_predict == test_y) / RF_predict.shape[0]
