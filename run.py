import argparse
import json
import os
from pathlib import Path
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np


def resolve_periodic_local_geometry(data, period, local_patch, local_stride):
    """Resolve zero-valued local geometry from training-only correlation evidence."""
    if local_patch < 0 or local_stride < 0:
        raise ValueError('periodic local patch/stride must be non-negative')
    if local_patch:
        return local_patch, local_stride or max(1, local_patch // 2)
    if local_stride:
        raise ValueError(
            'periodic_local_stride must be 0 when periodic_local_patch is auto (0)'
        )

    evidence_path = (
        Path(__file__).resolve().parent
        / 'logs'
        / 'graphmamba_local_scale'
        / f'{data}_local_scale.json'
    )
    if not evidence_path.exists():
        raise FileNotFoundError(
            f'missing training-derived local-scale evidence: {evidence_path}; '
            f'run scripts/derive_graphmamba_local_scale.py --dataset {data}'
        )
    evidence = json.loads(evidence_path.read_text())
    evidence_period = int(evidence['period'])
    if evidence_period != period:
        raise ValueError(
            f'local-scale evidence period {evidence_period} does not match '
            f'periodic_period {period}'
        )
    primary = evidence['primary']
    return int(primary['selected_patch']), int(primary['selected_stride'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--irpa_revise_len', type=int, default=96,
                        help='refined recent-window length for IRPA comparisons')
    parser.add_argument('--irpa_topk', type=int, default=3,
                        help='number of similar historical patches used by IRPA')
    parser.add_argument('--timerole_hidden_dim', '--cmrhm_hidden_dim',
                        dest='timerole_hidden_dim', type=int, default=32,
                        help='hidden dimension of the TimeRole history-correction branch')
    parser.add_argument('--timerole_memory_pool', '--cmrhm_memory_pool',
                        dest='timerole_memory_pool', type=int, default=16,
                        help='old-history average-pooling width for TimeRole')
    parser.add_argument('--timerole_recent_len', '--cmrhm_recent_len',
                        dest='timerole_recent_len', type=int, default=96,
                        help='recent-window length used by TimeRole and its strict control')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--test_after_train', type=int, choices=[0, 1], default=1,
                        help='run the test split after training; set to 0 during hyperparameter search')
    parser.add_argument('--evaluation_split', choices=['test', 'val'], default='test',
                        help='split used by an is_training=0 checkpoint evaluation')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2021, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GraphMamba
    parser.add_argument('--stride', type=int, default=8, help='GraphMamba long-patch stride')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_version', type=int, choices=[1, 2], default=1,
                        help='GraphMamba Mamba implementation: 1=Mamba-1, 2=Mamba2')
    parser.add_argument('--mamba_headdim', type=int, default=0,
                        help='Mamba2 head dimension; 0 selects GraphMamba default')
    parser.add_argument('--mamba_bidirectional', type=int, choices=[0, 1], default=1,
                        help='use forward and backward temporal Mamba branches')
    parser.add_argument('--use_graph', type=int, choices=[0, 1], default=1,
                        help='enable the GraphMamba graph branch')
    parser.add_argument('--use_time_mamba', type=int, choices=[0, 1], default=1,
                        help='enable the GraphMamba temporal branch')
    parser.add_argument('--use_patch', type=int, choices=[0, 1], default=1,
                        help='enable GraphMamba dual-scale patching')
    parser.add_argument('--use_decomp', type=int, choices=[0, 1], default=1,
                        help='enable GraphMamba moving-average decomposition')
    parser.add_argument('--dual_scale_scan_mode',
                        choices=['auto', 'joint', 'independent_shared', 'periodic_aligned'],
                        default='auto',
                        help='auto uses validated periodic mode on hourly ETT and independent scans elsewhere')
    parser.add_argument('--periodic_period', type=int, default=24,
                        help='training-derived stable period used by periodic_aligned')
    parser.add_argument('--periodic_local_patch', type=int, default=0,
                        help='within-period local patch length; 0 loads training-derived correlation scale')
    parser.add_argument('--periodic_local_stride', type=int, default=0,
                        help='local stride; 0 uses derived stride or half an explicit patch')
    parser.add_argument('--periodic_period_stride', type=int, default=12,
                        help='complete-period patch stride')
    parser.add_argument('--periodic_use_adapter', type=int, choices=[0, 1], default=1,
                        help='enable zero-initialized scale-conditioned input adaptation')
    parser.add_argument('--period_norm_factor', type=int, choices=[1, 4, 6], default=1,
                        help='native samples per physical hour for GraphMambaPeriodNorm')
    parser.add_argument('--period_norm_recent_len', type=int, default=96,
                        help='native-resolution recent window for GraphMambaPeriodNorm')
    parser.add_argument('--graph_alpha', type=float, default=0.3,
                        help='static graph weight in the static/adaptive graph blend')
    parser.add_argument('--graph_top_k', type=int, default=2,
                        help='number of static neighbors per variable')
    parser.add_argument('--graph_sample_size', type=int, default=2000,
                        help='training samples used to estimate the static graph')
    parser.add_argument('--graph_sample_method', choices=['uniform', 'random', 'recent'],
                        default='uniform', help='static graph sampling strategy')
    parser.add_argument('--static_graph_mode', choices=['weighted', 'binary'],
                        default='weighted', help='retain static edge weights or only topology')
    parser.add_argument('--static_graph_only', type=int, choices=[0, 1], default=0,
                        help='disable the adaptive graph and use only the static graph')
    parser.add_argument('--graph_cache', type=int, choices=[0, 1], default=0,
                        help='cache the generated static adjacency as a .npy file')
    parser.add_argument('--gc_graph_dim', type=int, default=16,
                        help='GraphMambaGC dynamic graph query/key dimension')
    parser.add_argument('--gc_temperature', type=float, default=1.0,
                        help='GraphMambaGC dynamic adjacency softmax temperature')
    parser.add_argument('--gc_residual_init', type=float, default=0.5,
                        help='initial graph-conditioning residual strength in (0, 1)')
    parser.add_argument('--gc_dynamic_graph', type=int, choices=[0, 1], default=1,
                        help='enable sample- and patch-adaptive variable graph')
    parser.add_argument('--gc_symmetric_graph', type=int, choices=[0, 1], default=1,
                        help='use shared projection for symmetric dynamic affinities')
    parser.add_argument('--gc_input_modulation', type=int, choices=[0, 1], default=1,
                        help='condition Mamba input tokens with graph context')
    parser.add_argument('--gc_direction_fusion', type=int, choices=[0, 1], default=1,
                        help='condition forward/backward Mamba fusion on graph context')
    parser.add_argument('--gc_parallel_residual', type=int, choices=[0, 1], default=1,
                        help='retain the original parallel graph residual beside graph conditioning')
    parser.add_argument('--af_hidden_dim', type=int, default=32,
                        help='GraphMambaAF reliability gate hidden dimension')
    parser.add_argument('--af_mode', choices=['local', 'variable_scale', 'variable_scale_residual', 'variable_scale_lowrank', 'residual_only'],
                        default='variable_scale_residual',
                        help='GraphMambaAF fusion/calibration mode')
    parser.add_argument('--af_rank', type=int, default=16,
                        help='rank of GraphMambaAF low-rank residual correction')
    parser.add_argument(
        '--timerole_old_intervention', '--cmrhm_old_intervention',
        dest='timerole_old_intervention',
        choices=['intact', 'batch_shuffle', 'temporal_shuffle', 'reverse',
                 'recent_mean', 'noise'],
        default='intact',
        help='evaluation-time intervention on the compressed old-history branch',
    )
    parser.add_argument('--timerole_noise_std', '--cmrhm_noise_std',
                        dest='timerole_noise_std', type=float, default=1.0,
                        help='normalized noise scale for the TimeRole noise intervention')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()

    if args.dual_scale_scan_mode == 'auto':
        args.dual_scale_scan_mode = (
            'periodic_aligned'
            if args.data in {'ETTh1', 'ETTh2'} and args.periodic_period < args.seq_len
            else 'independent_shared'
        )
    if args.model == 'GraphMamba' and args.dual_scale_scan_mode == 'periodic_aligned':
        args.periodic_local_patch, args.periodic_local_stride = (
            resolve_periodic_local_geometry(
                args.data,
                args.periodic_period,
                args.periodic_local_patch,
                args.periodic_local_stride,
            )
        )

    # Apply the requested experiment seed after parsing so --seed controls
    # model initialization, data shuffling, NumPy, and CUDA consistently.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            if args.model in {'GraphMamba', 'GraphMambaPeriodNorm', 'GraphMambaGC', 'GraphMambaAF', 'GraphMambaSD', 'GraphMambaGF', 'GraphMambaRG'}:
                setting += f'_patch{args.patch_len}_st{args.stride}_ds{args.d_state}' \
                           + f'_mv{args.mamba_version}_bi{args.mamba_bidirectional}' \
                           + f'_ga{args.graph_alpha}_gk{args.graph_top_k}'
                if args.model == 'GraphMamba':
                    if args.dual_scale_scan_mode == 'periodic_aligned':
                        setting += f'_smP_p{args.periodic_period}s{args.periodic_period_stride}' \
                                   + f'_l{args.periodic_local_patch}s{args.periodic_local_stride}' \
                                   + f'_a{args.periodic_use_adapter}'
                    else:
                        setting += f'_sm{args.dual_scale_scan_mode}'
                if args.model == 'GraphMambaPeriodNorm':
                    setting += f'_pnf{args.period_norm_factor}_r{args.period_norm_recent_len}' \
                               + f'_p{args.periodic_period}s{args.periodic_period_stride}' \
                               + f'_l{args.periodic_local_patch}s{args.periodic_local_stride}' \
                               + f'_a{args.periodic_use_adapter}'
            if args.model == 'GraphMambaGC':
                setting += f'_gd{args.gc_graph_dim}_gt{args.gc_temperature}' \
                           + f'_gr{args.gc_residual_init}_dyn{args.gc_dynamic_graph}' \
                           + f'_sym{args.gc_symmetric_graph}' \
                           + f'_im{args.gc_input_modulation}_df{args.gc_direction_fusion}' \
                           + f'_pr{args.gc_parallel_residual}'
            if args.model == 'GraphMambaAF':
                setting += f'_af{args.af_mode}_h{args.af_hidden_dim}_r{args.af_rank}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.test_after_train:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            else:
                print('>>>>>>>test skipped (validation-only run)<<<<<<<<<<<<<<<<<<<<<<<<')
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        if args.model in {'GraphMamba', 'GraphMambaPeriodNorm', 'GraphMambaGC', 'GraphMambaAF', 'GraphMambaSD', 'GraphMambaGF', 'GraphMambaRG'}:
            setting += f'_patch{args.patch_len}_st{args.stride}_ds{args.d_state}' \
                       + f'_mv{args.mamba_version}_bi{args.mamba_bidirectional}' \
                       + f'_ga{args.graph_alpha}_gk{args.graph_top_k}'
            if args.model == 'GraphMamba':
                if args.dual_scale_scan_mode == 'periodic_aligned':
                    setting += f'_smP_p{args.periodic_period}s{args.periodic_period_stride}' \
                               + f'_l{args.periodic_local_patch}s{args.periodic_local_stride}' \
                               + f'_a{args.periodic_use_adapter}'
                else:
                    setting += f'_sm{args.dual_scale_scan_mode}'
            if args.model == 'GraphMambaPeriodNorm':
                setting += f'_pnf{args.period_norm_factor}_r{args.period_norm_recent_len}' \
                           + f'_p{args.periodic_period}s{args.periodic_period_stride}' \
                           + f'_l{args.periodic_local_patch}s{args.periodic_local_stride}' \
                           + f'_a{args.periodic_use_adapter}'
        if args.model == 'GraphMambaGC':
            setting += f'_gd{args.gc_graph_dim}_gt{args.gc_temperature}' \
                       + f'_gr{args.gc_residual_init}_dyn{args.gc_dynamic_graph}' \
                       + f'_sym{args.gc_symmetric_graph}' \
                       + f'_im{args.gc_input_modulation}_df{args.gc_direction_fusion}' \
                       + f'_pr{args.gc_parallel_residual}'
        if args.model == 'GraphMambaAF':
            setting += f'_af{args.af_mode}_h{args.af_hidden_dim}_r{args.af_rank}'

        if args.evaluation_split == 'val':
            print('>>>>>>>evaluating validation checkpoint : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.evaluate_checkpoint(setting, flag='val')
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
