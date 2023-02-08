###### Results vary depending on the appropriate size and dimensions of the
# original image and annotation being transformed. Best to repeat multiple 
#times

import sys
import os

header = str(os.getcwd())
sub_module = header + r'\Documents\GitHub'
full_path = sub_module + r'\PyIR\src'
sys.path.append(full_path)

import pyir_image as pir_im


image_tool = pir_im.PyIR_Image()

Original_image = image_tool.import_annotations(r'C:\Users\Dougal\OneDrive - The University of Manchester\Pictures\Example_Original.png')
Warped_image =  image_tool.import_annotations(r'C:\Users\Dougal\OneDrive - The University of Manchester\Pictures\Example_warped.png')

image_tool.disp_image(Original_image)
image_tool.disp_image(Warped_image)

### generate control points for affine transform

transform = pir_im.PyIR_Image()

transform.control_point_selection(image_tool.rgb2gray(Warped_image), image_tool.rgb2gray(Original_image))


#apply affine transform
affined = transform.affine_transform(Warped_image, transform.source_coords,
                                     transform.desti_coords)

image_tool.disp_image(affined)

#crop image to desired shape and match sizes
affined = transform.crop_to_click(affined, Original_image)
affined = transform.match_size(affined, Original_image)
image_tool.disp_image(affined)


