import argparse
def parser_add_main_args(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='../../CLIP_finetune/')
    parser.add_argument('--text_dir_1',type=str,default="../../dataset/MarKG/entity2text.txt")
    parser.add_argument('--text_dir_2',type=str,default="../../dataset/MarKG/entity2textlong.txt")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_step', type=int,
                        default=1000, help='how often to save model')
    parser.add_argument('--load_step', type=int, default=0)

    # hyper_parameter for model arch and training
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--accumulate_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--structure',type=int,default=0)
    parser.add_argument('--load_mask',type=bool,default=False)
    parser.add_argument('--pretrain_length',type=int,default=512)
    parser.add_argument('--k_shot',type=int,default=64)

