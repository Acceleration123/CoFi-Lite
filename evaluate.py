import os
from omegaconf import OmegaConf


def main(args):
    config = OmegaConf.load(args.config)
    enh_folder = config.network.enh_folder
    
    if args.metric == 'dnsmos':
        os.system(
            ('python ./evaluation/calculate_nonintrusive_dnsmos.py '
                f'--inf_scp {enh_folder}/inf.scp '
                f'--output_dir {enh_folder}/scoring_dnsmos '
                '--device cuda '
                '--job 1 '
                '--convert_to_torch True '
                '--primary_model ./DNSMOS/DNSMOS/sig_bak_ovr.onnx '
                '--p808_model ./DNSMOS/DNSMOS/model_v8.onnx'
            )
        )    
    elif args.metric == 'intrusive':
        os.system(
            ('python ./evaluation/calculate_intrusive_se_metrics.py '
             f'--ref_scp {enh_folder}/ref.scp '
             f'--inf_scp {enh_folder}/inf.scp '
             f'--output_dir {enh_folder}/scoring_intrusive '
             '--nj 8 '
             '--chunksize 1000'
            )
        )

    else:
        raise ValueError
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, help="Metric to be calculated")
    parser.add_argument('--config', default='configs/cfg_infer.yaml')
    parser.add_argument('--device', default='1')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
