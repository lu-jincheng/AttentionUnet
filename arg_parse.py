import argparse


def get_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='cfgs.yml')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--label_names', type=str, default=None)
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--tf_record_prefix', type=str, default=None)
    parser.add_argument('--img_type', type=str, default=None)
    parser.add_argument('--img_size', default=None, type=int, nargs=2, help='image size of input')
    parser.add_argument('--dataset1', default=None, type=int, nargs=61, help='image size of input')
    parser.add_argument('--dataset2', default=None, type=int, nargs=61, help='image size of input')
    parser.add_argument('--dataset3', default=None, type=int, nargs=62, help='image size of input')
    parser.add_argument('--dataset4', default=None, type=int, nargs=62, help='image size of input')
    parser.add_argument('--channels', type=int, default=None)
    parser.add_argument('--train_tfrecord_path', type=str, default=None)
    parser.add_argument('--test_tfrecord_path', type=str, default=None)
    parser.add_argument('--net_type', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--opt', type=str, default=None, choices=('adam', 'momentum'))
    parser.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('-lr', default=None, type=float, help='learning rate')
    # parser.add_argument('-ld', default=None, type=float, help='learning rate decay')

    return parser.parse_args()
