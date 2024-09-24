import subprocess
import os


print(f"Current working directory: {os.getcwd()}")
if not os.getcwd().endswith("vdisnet"):
    os.chdir("/home/falcetta/0_PhD/sparse_var/vdisnet")
    print(f"Changed working directory: {os.getcwd()}")


code_dims = [128, 256, 512, 1024]
nums_iter_LISTA = [1,3,5,10]

# Repeat for each parameter combination

for code_dim in code_dims:
    for num_iter_LIST in nums_iter_LISTA:
        print(f"Testing code_dim: {code_dim} and num_iter_LISTA: {num_iter_LIST}")
        pass

# Construct the command as a list of arguments
command = [
    "python", "-u", "main.py",
    "--batch_size", "200",
    "--code_dim", "128",
    "--code_reg", "1",
    "--cuda",
    "--dataset", "VPATCHES",
    "--decoder", "linear_dictionary",
    "--encoder", "lista_encoder",
    "--epochs", "200", # default: 200
    "--FISTA",
    "--hidden_dim", "0",
    "--hinge_threshold", "0.5",
    "--im_size", "32", # If patch_size > im_size, then patch is resized to im_size !!!!
    "--lrt_D", "0.001",
    "--lrt_E", "0.0003",
    "--lrt_Z", "1",
    "--n_steps_inf", "200",
    "--n_test_samples", "10000",
    "--n_training_samples", "55000",
    "--n_val_samples", "5000",
    "--noise", "[1]",
    "--norm_decoder", "1",
    "--num_iter_LISTA", "3",
    "--num_workers", "4",
    "--patch_size", "0",
    "--positive_ISTA",
    "--seed", "31",
    "--sparsity_reg", "0.005",
    "--stop_early", "0.001",
    "--train_decoder",
    "--train_encoder",
    "--use_Zs_enc_as_init",
    "--variance_reg", "0",
    "--weight_decay_D", "0",
    "--weight_decay_E", "0",
    "--weight_decay_E_bias", "0",
    "--weight_decay_D_bias", "0"
]

# Execute the command
subprocess.run(command)
