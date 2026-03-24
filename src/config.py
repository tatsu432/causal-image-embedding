num_seeds = 10
n_train_fMNIST = 1000
n_test_fMNIST = 1000
# d_x
dim_covariate = 3
# d_x_image
dim_covariate_image = 10
# d_p_image
dim_post_treatment = 20

### AUTOENCODER CONFIGURATION ###
batch_size_autoencoder = 256

### CAUSAL INFERENCE CONFIGURATION ###
# Specify the sample size for the training and test dataset
# n_train
trainig_sample_size = 5000
# n_test
test_sample_size = 5000

# Define the batch size for causal inference
batch_size_causal_embedding = 256

# Define the learning rate, number of epochs, and weight decay for the embedding net
lr_embed = 1e-3
epochs_embed = 10
weight_decay_embed = 1e-5

# Define the dimension of the embedding of the image covariate and post-treatment
dim_covariate_image_embed = dim_covariate_image
dim_post_treatment_embed = dim_post_treatment

dim_covariate_image_embed_naive = dim_covariate_image_embed + dim_post_treatment_embed


### OTHER CONFIGURATION ###
print_loss = False
print_result_per_seed = False
display_image = True
