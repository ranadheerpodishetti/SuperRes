#Author: Geetha Doddapaneni Gopinath


#SRCNN Values
scale_factor = 2 #Can be changed based on the upscaling factor 
num_dimensions = 3
num_features = 32
activation_maps = scale_factor ** num_dimensions  #Number of  dimensions modified, hence number of activation maps needed is 8
kernel = 3 #(3,3,3)
stride = 1 
