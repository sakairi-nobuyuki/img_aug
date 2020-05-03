###  max number of augumentation
max_n_aug = 10  

###  parameters of rotation
prob_like_rot = 10
rotation_deg = [-5, -2, 2, 5]

### parameters of offset
prob_like_offset = 5
offset_dx =    [-20, -10, 0, 10, 20]
offset_dy =    [-20, -10, 0, 10, 20]

### parameters of blur
prob_like_blur = 3
k_blur = [1, 2, 3]

### parameters of sharpness
prob_like_sharpness = 2
A_sharpness = [[[0, -1, 0], [-1, 5, -1], [0, -1, 0]], [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]

### parameters of colour tone 
prob_like_colour_tone = 5
s_mag_list = [0.9, 0.8]   ## colour tone magnitude of saturation
v_mag_list = [0.9, 0.8]   ## colour tonemagnitude of value


###  parameters of contrast
prob_like_contrast = 5
alpha_list = [0.6, 1.0, 1.2, 1.4]   ### contrast
gamma_list = [-20, 0.0, 20, 50]       ### contrast


### parameters of Gaussian noize
prob_like_noize = 5
sigma_list = [1, 2, 5, 10]    ### gaussian noize