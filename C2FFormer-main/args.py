import argparse

def get_args():
    parser = argparse.ArgumentParser(description='STSGM')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--path', type=str, default='generated_datasets/italy_air_quality_rate01_step96_point')
    parser.add_argument('--MAR', type=bool, default=False,help='missing at random') 
    parser.add_argument('--onepiece', type=bool, default=False) 
    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--name', type=str, default='italy', help='dataset name')
    parser.add_argument('--rate', type=str, default='_rate0.1')
    parser.add_argument('--mini', type=bool, default=True)
    parser.add_argument('--afuse', type=bool, default=True)
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--tlr', type=float, default=1e-4)
    parser.add_argument('--fp16', type=int, default=1)
    parser.add_argument('--tclip', type=float, default=2)
    parser.add_argument('--ac', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--embed_dim',type=int, default=7, help='embedding dimension, same as input dimension')
    parser.add_argument('--num_heads', type=int, default=7, help='same as embed_dim')
    parser.add_argument('--depth', type=int, default=128)
    parser.add_argument('--result_pic_path', type=str, default='result_pic')
    args = parser.parse_args()
    args.result_pic_path = args.result_pic_path + args.rate + '/' + args.name
    args.tlr = args.tlr * args.batch_size / 256 
    
    if args.name == 'electricity':
        args.embed_dim = 370
        args.num_heads = 370
    if args.name == 'beijing':
        args.embed_dim = 132
        args.num_heads = 132
    if args.name == 'italy':
        args.embed_dim = 13
        args.num_heads = 13
    if args.name == 'pedestrian':
        args.embed_dim = 1
        args.num_heads = 1
    if args.name == 'physionet_2012':
        args.embed_dim = 35
        args.num_heads = 35
    if args.name == 'pems':
        args.embed_dim = 862
        args.num_heads = 862
    return args 
