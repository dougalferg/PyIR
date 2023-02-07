"""
    .. module:: PyIR_Mask
       :platform: Windows
       :synopsis: Mask object for 

    .. moduleauthor:: Dougal Ferguson <dougal.ferguson@manchester.ac.uk>

"""

import numpy as np
import pyir_spectralcollection, pyir_image, pyir_pca


class PyIR_Mask:
    """Doug_Mask is a child class from Doug_Toolbox() used mainly for the 
    creation of logic masks to be used for data extraction. This class
    uses super() inheritance to take all functions from Doug_Toolbox().

    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)


    def create_mask(self, data, test, value1=[], value2 =[]):  
        """Creates a logic mask based on inputted numerical tests and values.
        test can be '==', '!=', '<', '>', '<=', '>=','between', 'inside', 
        'within', 'outside', 'not_within'. Can handle 2D input arays in either
        [n,m] or [n*m, 0] form.
        Can handle 3D length rgb arrays of [n,m,[r,g,b]].
        
        :param data: Data to apply test to. In example the return from object.area_between(x, y) function.
        :type data: numpy.array
        :param test: Logical test to apply to the data.
        :type test: str
        :param value1: Value of the test argument.
        :type value1: int | float
        :param value2: Additional test argument value.
        :type value2: int | float
        
        :returns: bool mask (numpy.array)
        
        """
        
        #test can be '==', '!=', '<', '>', '<=', '>=',
        #'between', 'inside', 'within', 'outside', 'not_within'
        #Input will either be an nxm array, or 1xn 
        
        if value2 != []:
            value1, value2 = pyir_spectralcollection.arg_size_checker(value1, value2)   
        if test == '==':
            mask = (data == value1)
        elif test == '!=':
            mask = (data != value1)
        elif test == '<':
            mask = (data < value1)
        elif test == '>':
            mask = (data > value1)
        elif test == '<=':
            mask = (data <= value1)
        elif test == '>=':
            mask = (data >= value1)
        elif test in ('between' , 'inside' , 'within'):
            #value 1 is always the upper
            mask1 = (data <= value1) 
            mask2 = (data >= value2)
            mask = self.combine_and_(mask1, mask2)
        elif test in ('outside' ,'not_within'):
            mask1 = (data > value1) 
            mask2 = (data < value2)
            mask = self.combine_and_(mask1, mask2)
        #Check if the mask has a 3 length rgb return and handle as needed
        if (mask.ndim == 3 and len(mask[0][1]) == 3):
            mask = np.all(mask[:][:] == True, 2)
        
        return mask
    
    def combine_or_(self, *args):
        """Combines masks using the conditional OR (||) argument. Assumption
        is that all inputs are of the same dimension/size.
        
        :param args: Logical mask inputs.
        :type args: numpy.array
        
        :returns: bool mask (numpy.array)
        
        """
            
        ubermask = np.full(args[0].shape, False)
        Input_was_2dim = False
        
        if args[0].ndim ==2:
            orig_dims, ubermask = pyir_spectralcollection.reshaper_2D(ubermask)
            Input_was_2dim = True
            
        for arg in args:
            if arg.ndim == 2:
                _, arg = pyir_spectralcollection.reshaper_2D(arg)
            ubermask = np.logical_or(ubermask, arg)
        
        if Input_was_2dim == True:
            ubermask = pyir_spectralcollection.reshaper_2D(ubermask, orig_dims)
        
        return ubermask
        
    def combine_and_(self, *args):
        """Combines masks using the conditional AND (&) argument. Assumption
        is that all inputs are of the same dimension/size.
        
        :param args: Logical mask inputs.
        :type args: numpy.array
        
        :returns: bool mask (numpy.array)
        
        """
        ubermask = np.full(args[0].shape, True)
        
        Input_was_2dim = False
        
        if args[0].ndim ==2:
            orig_dims, ubermask = pyir_spectralcollection.reshaper_2D(ubermask)
            Input_was_2dim = True
        
        for arg in args:
            if arg.ndim == 2:
                _, arg = self.reshaper_2D(arg)
            ubermask = ubermask*arg
            
        if Input_was_2dim == True:
            ubermask = self.reshaper_2D(ubermask, orig_dims)
        
        return ubermask
    
    def combine_not_(self, *args):
        """Creates a mask that is not (!=) the input mask(s). Assumption
        is that all inputs are of the same dimension/size.
        
        :param args: Logical mask inputs.
        :type args: numpy.array
        
        :returns: bool mask (numpy.array)
        
        """
        ubermask = self.combine_or_(*args)
        
        return (np.absolute((ubermask-1))).astype(bool)
    
    def remove_edge(self, Mask, Pixels=1):
        """Alters a mask to remove a user defined number of pixels from the 
        borders between bools within a mask. Usage is to try to avoid sampling
        from tissue edges which could confuse trained models. Inputs must be
        in a 2-D form [n,m].
        
        :param Mask: Logical mask inputs.
        :type Mask: array of bool
        :param Pixels: Numper of pixels to remove inwards.Default = 1
        :type Pixels: int
        
        :returns: bool mask (Array of bool)
        
        """
        
        if Mask.ndim != 2:
            print("Mask dimensions incorrect, please input a 2D mask.")
        else:  
            output = np.full(Mask.shape, False)
            for y in np.arange(Pixels, (len(output[:,0])-Pixels)):
                for x in np.arange(Pixels,(len(output[0,:])-Pixels)):
                    if (Mask[y,x] == True):
                        test = []
                        for n in np.arange(0, Pixels):
                            test.append(Mask[y+(n+1), x-(n+1)])
                            test.append(Mask[y+(n+1), x])
                            test.append(Mask[y+(n+1), x+(n+1)])
                            test.append(Mask[y, x-(n+1)])
                            test.append(Mask[y, x+(n+1)])
                            test.append(Mask[y-(n+1), x-(n+1)])
                            test.append(Mask[y-(n+1), x])
                            test.append(Mask[y-(n+1), x+(n+1)])
                            if sum(test) != len(test):
                                break
                       
                        if sum(test) == len(test):
                            output[y,x] = True
                                           
            return output                      
                               