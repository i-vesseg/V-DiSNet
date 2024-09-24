import argparse
import collections
import json
import os
import random
import string
import time
from importlib import import_module

import torch
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

from utils import PSNR, MSE, normalize_kernels, L0,\
    set_random_seed, np, Gauge, my_iterator, my_iterator_val, ISTA, hinge, compute_energy, \
    sqrt_var, dewhiten, log_viz, anneal_learning_rate, \
    print_final_training_msg, print_final_eval_msg, mk_dir

from eval_denoising import eval_denoising

def define_args():
    # Define arguments
    parser = argparse.ArgumentParser(description='Multi-layer Sparse Coding with Variance Regularization.')

    # Experiment details
    parser.add_argument('--name', type=str, default='',
                        help='Name of experiment.')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='Random seed.')
    parser.add_argument('--outdir', default='./results/', type=str,
                        help='Path to the directory that contains the outputs.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to run code on GPU (default: run on CPU).')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers.')
    # Data processing
    parser.add_argument('--batch_size', type=int, default=250, metavar='N',
                        help='Mini-batch size.')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='Path to the directory that contains the data.')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | imagenet_LCN).')
    parser.add_argument('--data_splits', default='./data', type=str,
                        help='Path to the directory that contains the data splits.')
    parser.add_argument('--n_training_samples', type=int, default=55000, metavar='N',
                        help='Number of training samples for the model.')
    parser.add_argument('--n_val_samples', type=int, default=5000, metavar='N',
                        help='Number of validation samples for the model.')
    parser.add_argument('--n_test_samples', type=int, default=10000, metavar='N',
                        help='Number of test samples for evaluating the model.')
    parser.add_argument('--im_size', type=int, default=28,
                        help='Image input size.')
    parser.add_argument('--patch_size', type=int, default=0,
                        help='Patch size to sample after rescaling to im_size (0 if no patch sampling).')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of times the model passes through all the training data.')
    # Decoder arguments
    parser.add_argument('--decoder', default='linear_dictionary', type=str,
                        help='Decoder architecture.')
    parser.add_argument('--pretrained_path_dec', default='', type=str,
                        help='Path to the state_dict of a pre-trained decoder.')
    parser.add_argument('--train_decoder', action='store_true', default=False,
                        help='Whether to train a decoder (default is not).')
    parser.add_argument('--code_dim', type=int, default=128, metavar='N',
                        help='Code dimension.')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='N',
                        help='Hidden dimension for multi-layer decoder.')
    parser.add_argument('--norm_decoder', type=float, default=0,
                        help='Radius of the sphere the decoder\'s columns are projected to. Default: no normalization.')
    parser.add_argument('--lrt_D', type=float, default=1e-4,
                        help='Learning rate for the decoder weights.')
    parser.add_argument('--weight_decay_D', type=float, default=0,
                        help='Weight decay for decoder optimizer.')
    parser.add_argument('--weight_decay_D_bias', type=float, default=0,
                        help='Weight decay to use on bias term in decoder.')
    parser.add_argument('--anneal_lr_D_freq', type=int, default=0,
                        help='How frequently to anneal the decoder\'s learning rate.')
    parser.add_argument('--anneal_lr_D_mult', type=float, default=0,
                        help='Multiplier for annealing the decoder\'s learning rate.')
    # Encoder arguments
    parser.add_argument('--encoder', default='lista_encoder', type=str,
                        help='Encoder architecture.')
    parser.add_argument('--pretrained_path_enc', default='', type=str,
                        help='Path to the state_dict of a pre-trained encoder.')
    parser.add_argument('--train_encoder', action='store_true', default=False,
                        help='Whether to train an encoder to predict codes from inference (default is not).')
    parser.add_argument('--num_iter_LISTA', type=int, default=0,
                        help='Number of LISTA iterations.')
    parser.add_argument('--lrt_E', type=float, default=1e-4,
                        help='Learning rate for the encoder weights.')
    parser.add_argument('--weight_decay_E', type=float, default=0,
                        help='Weight decay for encoder\'s parameters.')
    parser.add_argument('--weight_decay_E_bias', type=float, default=0,
                        help='Weight decay for encoder\'s bias.')
    parser.add_argument('--anneal_lr_E_freq', type=int, default=0,
                        help='How frequently to anneal the encoder\'s learning rate.')
    parser.add_argument('--anneal_lr_E_mult', type=float, default=0,
                        help='Multiplier for annealing the encoder\'s learning rate.')
    # Inference arguments
    parser.add_argument('--sparsity_reg', type=float, default=1e-3,
                        help='Sparsity term for codes during training.')
    parser.add_argument('--lrt_Z', type=float, default=1,
                        help='Learning rate for sparse codes calculation.')
    parser.add_argument('--positive_ISTA', action='store_true', default=False,
                        help='Whether to constrain ISTA to positive values.')
    parser.add_argument('--FISTA', action='store_true', default=False,
                        help='Whether to use a faster version of ISTA.')
    parser.add_argument('--n_steps_inf', type=int, default=200, metavar='N',
                        help='Number of inference iterations for computing each code.')
    parser.add_argument('--stop_early', type=float, default=1e-3,
                        help='Tolerance for early stopping during (F)ISTA.')
    parser.add_argument('--use_Zs_enc_as_init', action='store_true', default=False,
                        help='Epoch from which to start using encoder\'s predictions as initial values for ISTA.')
    parser.add_argument('--Zs_init_val', type=float, default=0,
                        help='Initial value to initialize codes for ISTA in non-linear model.')
    parser.add_argument('--variance_reg', type=float, default=0,
                        help='Weight of regularization term: squared hinge on std of latent components.')
    parser.add_argument('--hinge_threshold', type=float, default=0.5,
                        help='Threshold in the hinge loss.')
    parser.add_argument('--code_reg', type=float, default=0,
                        help='Coefficient for energy coming from distance to the encoder\'s predictions.')
    # Evaluation
    parser.add_argument('--noise', type=str, default='[]',
                        help='List with levels of noise for denoising evaluation.')
    # Parse arguments
    args = parser.parse_args()
    args.date_exp = time.strftime("%Y-%m-%d/%H-%M-%S")
    print(f"Date: {args.date_exp}")
    args.binary_output = True
    return args

def train_decoder_step(decoder, y, y_mean, y_std, Zs, optimizer_dec, args):
    # Decoder
    decoder.train()

    # Reconstruction loss
    y_hat = decoder(Zs)
    rec_loss_y = MSE(y, y_hat, reduction='mean')

    # Backward pass
    optimizer_dec.zero_grad()
    decoder.zero_grad()
    rec_loss_y.backward()
    optimizer_dec.step()

    # Compute PSNR
    #y_hat = binarize_prediction(y, y_hat) if args.binary_output else y_hat
    psnr = PSNR(y, y_hat, args.dataset, y_mean, y_std, binary_output=args.binary_output)

    # Normalize decoder weights
    if args.norm_decoder > 0:
        normalize_kernels(decoder, args.norm_decoder)

    # Output dictionary
    output = {'rec_loss_y': rec_loss_y.detach(),
              'y_hat': y_hat.detach(),
              'psnr': psnr}
    return output

def train_encoder_step(encoder, optimizer_enc, Zs_enc, Zs_inf):
    # Loss from codes
    rec_loss_code = MSE(Zs_enc, Zs_inf, reduction='mean')

    # Update encoder
    encoder.zero_grad()
    optimizer_enc.zero_grad()
    rec_loss_code.backward()
    optimizer_enc.step()

    output = {'rec_loss_code': rec_loss_code.detach()}
    return output

def main():
    
    print("----- START -----")
    # Get arguments
    print("--- Step 0: Get arguments")
    args = define_args()

    # Create directory structure
    outdir = lambda dirname: os.path.join(args.outdir, args.date_exp if dirname != 'final' else '', dirname)
    print(f"Output directory: {outdir('')}")
    
    print("--- Step 1: Create directory structure")
    mk_dir(args.outdir)
    mk_dir(outdir('checkpoints'))
    mk_dir(outdir('logs'))
    mk_dir(outdir('imgs'))
    mk_dir(outdir('final'))

    # Experiment name
    if len(args.name) == 0:
        timestamp = str(int(time.time()))
        args.unq = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
        args.name = '{}_{}_fn_{}_lrtZ_{}_lrtD_{}_ns_{}_sp_{}_s{}_{}_{}'.format(
            args.decoder, 
            args.dataset,
            args.norm_decoder,
            args.lrt_Z,
            args.lrt_D,
            args.n_steps_inf,
            args.sparsity_reg,
            args.seed,
            timestamp,
            args.unq)
    else:
        timestamp = args.name.split('_')[-2]
        args.unq = args.name.split('_')[-1]
    print('\n---- Experiment name : {} ----\n'.format(args.name))

    # Experiment directory for saving visualizations
    img_dir = outdir('imgs')
    mk_dir(img_dir)
    print(f"Visualizations will be saved in {img_dir}")
    
    # create a txt file to save the name of the experiment in img_dir
    with open(os.path.join(img_dir, 'experiment_name.txt'), 'w') as f:
        f.write(args.name)
        f.write(f"\n Date: {args.date_exp}")
        f.write(f"\n Decoder: {args.decoder}")
        f.write(f"\n Dataset: {args.dataset}")
        f.write(f"\n Norm decoder: {args.norm_decoder}")
        f.write(f"\n Learning rate Z: {args.lrt_Z}")
        f.write(f"\n Learning rate D: {args.lrt_D}")
        f.write(f"\n Number of steps inference: {args.n_steps_inf}")
        f.write(f"\n Sparsity reg: {args.sparsity_reg}")
        f.write(f"\n Seed: {args.seed}")
        f.write(f"\n Timestamp: {timestamp}")
        f.write(f"\n Unique: {args.unq}")
    
    # Print and save experiment specs
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    json_file = open(os.path.join(outdir('final'), args.unq + '.json'), "w")
    json_file.write(json.dumps(args.__dict__, sort_keys=True, indent=4))
    json_file.close()

    # More logistics
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {device}")
    
    whitening = args.dataset == 'imagenet_LCN'
    print(f"Whitening: {whitening}")
    args.noise = eval(args.noise)
    head = f"m\ttime\tunq\tenc\tdata\ttr_D\ttr_E\tFISTA\tmd\ts\tcd\thd\tsp\tvar\tht\tnd\twd_B\t" \
           f"init\tlrt_Z\tlrt_D\tanf\tanr\twd_D\tlrt_E\titer\tuse_enc\twdE\twdE_b\tcd_reg\tsteps\t" \
           f"L0_Z\tL0_H\tPSNR\tep\tev\tL0_orig\torig\tnoisy_im\tL0_noisy\tnoisy_rec"
    msg_pre = f"{args.decoder}\t{timestamp}\t{args.unq}\t{args.encoder}\t{args.dataset}\t" \
              f"{int(args.train_decoder)}\t{int(args.train_encoder)}\t{int(args.FISTA)}"
    msg_post = f"{args.seed}\t{args.code_dim}\t{args.hidden_dim}\t{args.sparsity_reg:.1e}\t{args.variance_reg}\t" \
               f"{args.hinge_threshold}\t{args.norm_decoder}\t{args.weight_decay_D_bias}\t{args.Zs_init_val}\t" \
               f"{args.lrt_Z}\t{args.lrt_D}\t{args.anneal_lr_D_freq}\t{args.anneal_lr_D_mult}\t{args.weight_decay_D}\t" \
               f"{args.lrt_E}\t{args.num_iter_LISTA}\t{int(args.use_Zs_enc_as_init)}\t{args.weight_decay_E}\t" \
               f"{args.weight_decay_E_bias}\t{args.code_reg}\t"

    # Tensorboard support. To run: tensorboard --logdir <args.outdir>/logs
    experiment_logs_dir = outdir('logs')
    mk_dir(experiment_logs_dir)
    writer = SummaryWriter(log_dir=experiment_logs_dir)

    # Random seed
    print("--- Step 2: Set random seed")
    set_random_seed(args.seed, torch, np, random, args.cuda)


    # -------------------------------------------------------------------------------------------------------------------------------#
    from patch_dataloader.load_patches import main as get_dataloader
    
    path_data = "/data/falcetta/brain_data"
    original_data_dir = os.path.join(path_data, "IXIJ/processed/numpy_renamed/train") 
    preprocessed_data_dir = os.path.join(path_data, "IXIJ/processed/patches_preprocessed_with_empty") 
    
    standardize_volume = True
    use_true_labels_train = True
    patch_size = 32
    
    dataloader_parameters = {
        "patch_size": patch_size,
        "validation_split": 0.2,
        "batch_size": args.batch_size,
        "normalize": True,
        "augmentation": False,
        "pixel_wise": True,
    }
    data_train, data_val = get_dataloader(preprocessed_data_dir, standardize_volume, use_true_labels_train, original_data_dir, **dataloader_parameters)
    args.n_channels = 1
         
    # Logistics: data
    print(f"\n\n Number of channels: {args.n_channels}\n\n")
    # Logistics: logging
    n_epochs_to_log_imgs = min(10, args.epochs)
    log_viz_interval = int(args.epochs / n_epochs_to_log_imgs)
    print(f"Log visualizations every {log_viz_interval} epochs")

    # Logistics: keep track of best training and validation models
    best_perf_tr = collections.defaultdict(lambda: None)
    best_perf_val = collections.defaultdict(lambda: None)
    idn = "_".join(args.name.split('_')[-2:]) # experiment name without the timestamp and unique identifier
    results_file = os.path.join(outdir('final'), idn + '.csv')
    if args.train_encoder + args.train_decoder == 0:
        results_file = os.path.join(outdir('final'), 'EVAL-only_' + idn + '.csv')

    # Decoder
    decoder = getattr(import_module('models.{}'.format(args.decoder)), 'Decoder')(args).to(device)
    if len(args.pretrained_path_dec) > 0:
        # Load pretrained decoder, turn off gradients if not training it
        print("--- Step 4: Load pretrained decoder, turn off gradients if not training it")
        decoder.load_pretrained(args.pretrained_path_dec, freeze=not(args.train_decoder))
    if args.train_decoder:
        # Normalize the decoder's columns, if randomly initialized
        print("--- Step 5: Normalize the decoder's columns, if randomly initialized")
        if args.norm_decoder > 0 and len(args.pretrained_path_dec) == 0:
            print(f"Normalizing decoder's columns to radius {args.norm_decoder}")
            normalize_kernels(decoder, args.norm_decoder)
    else:
        # If not training the decoder, put it in eval() mode and remove gradient tracking
        decoder.eval()
        decoder.requires_grad_(False)
        # If not training the decoder and there is no pre-trained decoder, save the random decoder (for eval)
        if len(args.pretrained_path_dec) == 0:
            args.pretrained_path_dec = outdir('checkpoints') + f'/DEC_random.pth'
            torch.save(decoder.state_dict(), args.pretrained_path_dec)

    # Encoder
    encoder = getattr(import_module('models.{}'.format(args.encoder)), 'Encoder')(args).to(device)
    if len(args.pretrained_path_enc) > 0:
        # Load pretrained encoder, turn off gradients if not training it
        print("--- Step 6: Load pretrained encoder, turn off gradients if not training it")
        encoder.load_pretrained(args.pretrained_path_enc, freeze=not(args.train_encoder))
    if not(args.train_encoder):
        # If not training the encoder, put it in eval() mode and remove gradient tracking
        encoder.eval()
        encoder.requires_grad_(False)
        # If not training the encoder and there is no pre-trained encoder, save the random encoder (for eval)
        if len(args.pretrained_path_enc) == 0:
            # Save the random encoder (for eval)
            print("--- Step 7: Save the random encoder (for eval)")
            args.pretrained_path_enc = outdir('checkpoints') + f'/ENC_random.pth'
            torch.save(encoder.state_dict(), args.pretrained_path_enc)

    def train(args):
        # Optimizer for decoder
        if args.train_decoder:
            optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.lrt_D, weight_decay=args.weight_decay_D)
            if args.decoder in ['one_hidden_decoder']:
                param_groups = [{'params': decoder.layer1.bias, 'weight_decay': args.weight_decay_D_bias},
                                {'params': decoder.layer1.weight},
                                {'params': decoder.layer2.weight}]
                optimizer_dec = torch.optim.Adam(param_groups, lr=args.lrt_D, weight_decay=args.weight_decay_D)

        # Optimizer for encoder
        if args.train_encoder:
            optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=args.lrt_E, weight_decay=args.weight_decay_E)
            if args.weight_decay_E_bias > 0:
                param_groups = [{'params': encoder.W.bias, 'weight_decay': args.weight_decay_E_bias},
                                {'params': encoder.W.weight},
                                {'params': encoder.S.weight}]
                optimizer_enc = torch.optim.Adam(param_groups, lr=args.lrt_E, weight_decay=args.weight_decay_E)

        # Gauge data
        gauge = Gauge()

        # Training loop
        print("--- Step 8: Training loop")
        train_iterator = my_iterator(args, data_train, log_viz_interval)
        for batch, batch_info, should in train_iterator:
            training = True
            epoch = batch_info['epoch']
            y = batch['X'].to(device)
            # if the image is all zeros, skip the iteration
            for bz in y:
                if torch.sum(bz) == 0:   
                    #save y as a numpy array
                    y_np = y.cpu().detach().numpy()
                    np.save(os.path.join(f"y_AA{epoch}.npy"), y_np)
                    print(y)
                    assert(), "AAA y is all zeros"
                    
            y_mean, y_std = None, None # whitening
            if args.dataset == 'imagenet_LCN':
                y_mean, y_std = batch['extra']
                y_mean, y_std = y_mean.to(device), y_std.to(device)

            # Track active code components
            if should['epoch_start']:
                Zs_comp_use = 0.

            # Encoder predictions
            if args.train_encoder:
                encoder.train() # Put encoder in train mode

            #print(f"Shape / MIN / MAX of y: {y.shape} / {y.min()} / {y.max()}")
            #print(f"UNIQUE")
            #assert()
            Zs_enc = encoder(y) # Encoder input: y
            #print(f"Zs_enc shape: {Zs_enc.shape}")
            #print(f"Zs_enc: {Zs_enc}")

            # Inference of the codes
            if args.n_steps_inf > 0:
                # Perform inference with (F)ISTA
                inference_output = ISTA(decoder, y, args.positive_ISTA, args.FISTA,
                      args.sparsity_reg, args.n_steps_inf, args.lrt_Z,
                      args.use_Zs_enc_as_init, Zs_enc,
                      args.variance_reg, args.hinge_threshold, args.code_reg,
                      args.stop_early, training, args.train_decoder)
                Zs = inference_output['Zs']
                #print(f"Zs shape: {Zs.shape}")
                #print(f"Zs: {Zs}")
                #print(f"Percentage of zero elements in Zs: {L0(Zs)}")
            else:
                # Amortized inference using the encoder's predictions
                Zs = Zs_enc.detach()

            # Decoder update
            if args.train_decoder:
                decoder_output = train_decoder_step(decoder, y, y_mean, y_std, Zs, optimizer_dec, args)
                y_hat = decoder_output['y_hat']
                rec_loss_y = decoder_output['rec_loss_y']
                psnr = decoder_output['psnr']
            else:
                with torch.no_grad():
                    y_hat = decoder(Zs)
                    rec_loss_y = MSE(y, y_hat, reduction='mean')
                    #y_hat = binarize_prediction(y_hat) if args.binary_output else y_hat
                    psnr = PSNR(y, y_hat, args.dataset, y_mean, y_std, binary_output=args.binary_output)

            # Encoder update
            if args.train_encoder and args.n_steps_inf > 0:
                encoder_output = train_encoder_step(encoder, optimizer_enc, Zs_enc, Zs)
                rec_loss_code = encoder_output['rec_loss_code']
            else:
                # Encoder will not be updated if encoder is not trained or there is no inference
                pass

            # Decoder stats
            gauge.add('rec_loss_y', rec_loss_y)
            gauge.add('PSNR', psnr)
            if args.decoder in ['linear_dictionary']:
                gauge.add('avg_col_norm', decoder.decoder.weight.data.norm(dim=0, p=2).mean())
            if args.decoder in ['one_hidden_decoder']:
                gauge.add('layer1_avg_col_norm', decoder.layer1.weight.data.norm(dim=0, p=2).mean())
                gauge.add('layer2_avg_col_norm', decoder.layer2.weight.data.norm(dim=0, p=2).mean())
                gauge.add('bias_norm_D', decoder.layer1.bias.data.norm())
                gauge.add('frac_0s_hidden_pre', decoder.frac_0s_hidden_pre_relu)
                gauge.add('frac_0s_hidden_post', decoder.frac_0s_hidden_post_relu)

            # Encoder stats
            if args.train_encoder and args.n_steps_inf > 0:
                gauge.add('rec_loss_Z', rec_loss_code)
            if args.encoder in ['lista_encoder']:
                gauge.add('bias_norm_E', encoder.W.bias.data.norm())

            # Code stats
            Zs_comp_use += (Zs.detach().abs() > 0).float().mean(0)
            hinge_loss = hinge(input=sqrt_var(Zs.detach()), threshold=args.hinge_threshold, reduction='mean')
            energy = compute_energy(y, y_hat, Zs,
                                    args.sparsity_reg, args.variance_reg, args.hinge_threshold, args.code_reg,
                                    Zs_enc)
            gauge.add('hinge_loss_Z', hinge_loss)
            gauge.add('energy', energy)
            gauge.add('frac_0s_Z', L0(Zs))
            gauge.add('max_Z', Zs.detach().abs().max())
            if args.n_steps_inf > 0:
                gauge.add('inf_steps', inference_output['inf_steps'])
                gauge.add('inf_time', inference_output['inference_time'])


            # End of epoch
            if should['epoch_end']:
                # Compute training metrics for the epoch
                #print("--- Compute training metrics for the epoch")
                train_stats = {}
                keys = list(gauge.cache.keys())
                for k in keys:
                    vals = torch.stack(gauge.get(k, clear=True))
                    v = torch.max(vals) if 'max' in k else torch.mean(vals)
                    writer.add_scalar(f'epoch_stats_training/{k}', v, epoch)
                    train_stats[k] = v
                train_stats['Zs_comp_use'] = Zs_comp_use / len(data_train)

                # Track best training performance (energy)
                if best_perf_tr['energy'] is None or best_perf_tr['energy'] > train_stats['energy']:
                    best_perf_tr['energy'] = train_stats['energy']
                    best_perf_tr['PSNR'] = train_stats['PSNR']
                    best_perf_tr['epoch'] = epoch
                    best_perf_tr['L0_Z'] = train_stats['frac_0s_Z']
                    if args.decoder in ['one_hidden_decoder']:
                        best_perf_tr['L0_H'] = train_stats['frac_0s_hidden_post']
                    if args.n_steps_inf > 0:
                        best_perf_tr['inf_steps'] = train_stats['inf_steps']

                # Validation
                #print("--- Validation + log visualizations")
                run_validation(encoder, decoder, epoch, args)

                # De-whiten
                if whitening:
                    print("--- De-whiten")
                    y = dewhiten(y, y_mean, y_std)
                    y_hat = dewhiten(y_hat.detach(), y_mean, y_std)

                # Log visualizations
                n_samples = 64
                #print("--- Log visualizations ---")
                img_dir_epoch = os.path.join(img_dir, f'ep_{epoch}/train')
                mk_dir(img_dir_epoch)
                log_viz(decoder, writer, n_samples, y, y_hat, Zs, train_stats, img_dir_epoch, args.decoder, args.dataset,
                        f'ep_{epoch}_train', log_all=True)

                # Save best model & useful viz
                if best_perf_val['epoch'] == epoch:
                    # Save encoder and decoder
                    if args.train_encoder:
                        torch.save(encoder.state_dict(), outdir('checkpoints') + f'/ENC_best.pth')
                    if args.train_decoder:
                        torch.save(decoder.state_dict(), outdir('checkpoints') + f'/DEC_best.pth')

                # Anneal lrt_D
                if args.train_decoder and args.anneal_lr_D_freq > 0:
                    anneal_learning_rate(optimizer_dec, epoch + 1, args.lrt_D, args.anneal_lr_D_mult,
                                         args.anneal_lr_D_freq)
                # Anneal lrt_E
                if args.train_encoder and args.anneal_lr_E_freq > 0:
                    anneal_learning_rate(optimizer_enc, epoch + 1, args.lrt_E, args.anneal_lr_E_mult,
                                         args.anneal_lr_E_freq)

            # Clean up memory
            del Zs, y, y_hat

        # Message (to help with analysis)
        print_final_training_msg(results_file, head, msg_pre, msg_post,
                                 args.noise, best_perf_tr, best_perf_val)

        writer.close()

    def run_validation(encoder, decoder, epoch, args):
        # The encoder's predictions are used as the Zs during validation
        training = False
        encoder.eval()
        decoder.eval()

        # Track metrics and logistics
        gauge = Gauge()
        Zs_comp_use = 0.

        # Data loop
        val_iterator = my_iterator_val(args, data_val, log_viz_interval, epoch)
        for batch, batch_info, should in val_iterator:
            y = batch['X'].to(device)
            # if the image is all zeros, skip the iteration
            if torch.sum(y) == 0:
                continue
            
            y_mean, y_std = None, None # whitening
            if args.dataset == 'imagenet_LCN':
                y_mean, y_std = batch['extra']
                y_mean, y_std = y_mean.to(device), y_std.to(device)

            # Encoder predictions
            with torch.no_grad():
                Zs_enc = encoder(y)

            # Compute the Zs from inference
            if args.n_steps_inf > 0:
                inference_output = ISTA(decoder, y, args.positive_ISTA, args.FISTA,
                                        args.sparsity_reg, args.n_steps_inf, args.lrt_Z,
                                        args.use_Zs_enc_as_init, Zs_enc,
                                        args.variance_reg, args.hinge_threshold, args.code_reg,
                                        args.stop_early, training, args.train_decoder)
                Zs_inf = inference_output['Zs']
            else:
                Zs_inf = Zs_enc

            # Validation stats
            with torch.no_grad():
                # Decoder stats
                y_hat = decoder(Zs_enc)
                rec_loss_y = MSE(y, y_hat, reduction='mean')
                #y_hat = binarize_prediction(y, y_hat) if args.binary_output else y_hat
                psnr = PSNR(y, y_hat, args.dataset, y_mean, y_std, binary_output=args.binary_output)


                # Encoder stats
                rec_loss_code = MSE(Zs_inf, Zs_enc, reduction='mean')

                # Code stats
                frac_0s = L0(Zs_enc)
                Zs_comp_use += (Zs_enc.detach().abs() > 0).float().mean(0)
                Zs_max = Zs_enc.detach().abs().max()
                energy = compute_energy(y, y_hat, Zs_enc,
                                        args.sparsity_reg, args.variance_reg, args.hinge_threshold, args.code_reg,
                                        Zs_enc)

                # Track stats
                gauge.add('rec_loss_Z', rec_loss_code)
                gauge.add('rec_loss_y', rec_loss_y)
                gauge.add('PSNR', psnr)
                gauge.add('frac_0s_Z', frac_0s)
                gauge.add('max_Z', Zs_max)
                gauge.add('energy', energy)
                if args.decoder in ['one_hidden_decoder']:
                    gauge.add(f'frac_0s_hidden_pre', decoder.frac_0s_hidden_pre_relu)
                    gauge.add(f'frac_0s_hidden_post', decoder.frac_0s_hidden_post_relu)

        # Log aggregate validation stats
        valid_stats = {}
        keys = list(gauge.cache.keys())
        for k in keys:
            vals = torch.stack(gauge.get(k, clear=True))
            v = torch.max(vals) if 'max' in k else torch.mean(vals)
            writer.add_scalar(f'epoch_stats_validation/{k}', v, epoch)
            valid_stats[k] = v
        valid_stats['Zs_comp_use'] = Zs_comp_use / len(data_val)

        # Track best validation performance
        if best_perf_val['energy'] is None or best_perf_val['energy'] > valid_stats['energy']:
            best_perf_val['energy'] = valid_stats['energy']
            best_perf_val['PSNR'] = valid_stats['PSNR']
            best_perf_val['epoch'] = epoch
            best_perf_val['L0_Z'] = valid_stats['frac_0s_Z']
            if args.decoder in ['one_hidden_decoder']:
                best_perf_val[f'L0_H'] = valid_stats['frac_0s_hidden_post']

        # De-whiten
        if whitening:
            y = dewhiten(y, y_mean, y_std)
            y_hat = dewhiten(y_hat.detach(), y_mean, y_std)

        # Log visualizations
        n_samples = 64
        if should['log_val_imgs']:
            #print("--- Log val visualizations ---")
            img_dir_val = os.path.join(img_dir, f'ep_{epoch}/val')
            mk_dir(img_dir_val)
            log_viz(decoder, writer, n_samples, y, y_hat, Zs_enc, valid_stats, img_dir_val, args.decoder, args.dataset,
                    viz_type=f'ep_{epoch}_val', log_all=False)
        if best_perf_val[f'epoch'] == epoch:
            #print("--- Log best val visualizations ---")
            img_dir_best_val = os.path.join(img_dir, 'BEST_VAL')
            mk_dir(img_dir_best_val)
            log_viz(decoder, writer, n_samples, y, y_hat, Zs_enc, valid_stats, img_dir_best_val, args.decoder, args.dataset,
                    viz_type='BEST_VAL', log_all=True)

    def run_eval_denoising(args):
        # Experiment identification
        idn = "_".join(args.name.split('_')[-2:]) # experiment name without the timestamp and unique identifier

        # Generate the arguments for the eval experiment
        for noise in args.noise:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.additive_noise = noise
            args_copy.name = f'ENC_{idn}_den_{noise}_{str(int(time.time()))}'
            # Path to encoder
            if args.train_encoder:
                args_copy.pretrained_path_enc = outdir('checkpoints') + f'/ENC_best.pth'
            # Path to decoder
            if args.train_decoder:
                args_copy.pretrained_path_dec = outdir('checkpoints') + f'/DEC_best.pth'
            eval_denoising(args_copy)
            print_final_eval_msg(results_file, msg_pre, msg_post, args_copy, best_perf_val)

    # Training
    if args.train_decoder or args.train_encoder:
        train(args)

    # Evaluation
    print("--- Denoising evaluation ---")
    run_eval_denoising(args)

if __name__ == '__main__':
    main()
