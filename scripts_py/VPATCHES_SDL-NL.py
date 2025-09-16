import subprocess
import os

print(f"Current working directory: {os.getcwd()}")
if not os.getcwd().endswith("vdisnet"):
    os.chdir("/home/falcetta/0_PhD/sparse_var/vdisnet")
    print(f"Changed working directory: {os.getcwd()}")


# Construct the command as a list of arguments
command = [
    "python", "-u", "main.py",
    "--batch_size", "250",
    "--code_dim", "128",
    "--code_reg", "1",
    "--cuda",
    "--dataset", "VPATCHES",
    "--decoder", "one_hidden_decoder", #LINEAR EQUIVALENT IN THE COMMENT
    "--encoder", "lista_encoder",
    "--epochs", "200", # default: 200
    "--FISTA",
    "--hidden_dim", "1024", ###### 0
    "--hinge_threshold", "0.5",
    "--im_size", "32",
    "--lrt_D", "0.001",
    "--lrt_E", "0.0001", ###### 0.0003
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
    "--sparsity_reg", "0.01", ###### 0.005 
    "--stop_early", "0.001",
    "--train_decoder",
    "--train_encoder",
    "--use_Zs_enc_as_init",
    "--variance_reg", "0",
    "--weight_decay_D", "0",
    "--weight_decay_E", "0",
    "--weight_decay_E_bias", "0",
    "--weight_decay_D_bias", "0.001" ###### 0
]

# Execute the command
subprocess.run(command)
