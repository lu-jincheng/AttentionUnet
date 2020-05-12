# coding: utf-8

import yaml
from main import *
from dataprovider import *
from make_tfrecords import *
import arg_parse


def cfg_parse(args, cfg):
    args_dict = args.__dict__
    args_key_list = [key for key in args_dict]
    cfg_key_list = [key for key in cfg]

    # init None args with cfg values
    undefined_arg_key = filter(lambda x, args_dict=args_dict: args_dict[x] is None, args_key_list)
    undefined_arg_key = filter(lambda x: x in cfg_key_list, undefined_arg_key)
    for key_name in undefined_arg_key:
        args_dict[key_name] = cfg[key_name]

    # add args which are not included in args parser
    uncontained_arg_key = filter(lambda x: not (x in args_key_list), cfg_key_list)
    for key_name in uncontained_arg_key:
        args_dict[key_name] = cfg[key_name]

    return args


def main():
    args = arg_parse.get_arg()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)
    f.close()
    args = cfg_parse(args, cfg)
    if args.loss == 'cross_entropy_log_dice':
        loss_kwargs = {'alpha': args.alpha}
    else:
        loss_kwargs = {}

    data = [args.dataset1, args.dataset2, args.dataset3, args.dataset4]
    test_num = args.testset
    testset = data[test_num - 1]
    trainset = []
    for i in range(4):
        if i != test_num - 1:
            trainset += data[i]
    tfrecord_train = TFrecord_Create_For_Unet(train_test='train',
                                              dataset=trainset,
                                              img_folder=args.img_dir,
                                              label_names=args.label_names,
                                              img_type=args.img_type,
                                              tf_record_pre_fix=args.tf_record_prefix,
                                              nx=args.img_size[0],
                                              ny=args.img_size[1]
                                              )

    tfrecord_test = TFrecord_Create_For_Unet(train_test='test',
                                             dataset=testset,
                                             img_folder=args.img_dir,
                                             img_type=args.img_type,
                                             label_names=args.label_names,
                                             tf_record_pre_fix=args.tf_record_prefix,
                                             nx=args.img_size[0],
                                             ny=args.img_size[1]
                                             )
    train_size = tfrecord_train.image_count
    test_size = tfrecord_test.image_count
    training_iters = int(math.ceil(train_size / args.batch_size))

    # Set up Dataprovider
    data_provider = Tfrecord_ImageDataProvider(
        train_tfrecord_path=args.train_tfrecord_path,
        test_tfrecord_path=args.test_tfrecord_path,
        channels=args.channels, train_batch_size=args.batch_size, test_batch_size=1,
        nx=args.img_size[0], ny=args.img_size[1])

    # Training
    net = Network(net_type=args.net_type, loss=args.loss, layers=5, features_root=64, channels=args.channels,
                  loss_kwargs=loss_kwargs)
    trainer = Trainer(net, data_provider=data_provider, batch_size=args.batch_size, validation_batch_size=1,
                      optimizer=args.opt, lr=args.lr, nx=args.img_size[0], ny=args.img_size[1], opt_kwargs={})
    trainer.train(output_path=args.model_path, log_path=args.log_path, prediction_path=args.pred_path,
                  training_iters=training_iters, epochs=args.n_epochs, test_size=test_size)
    _d, _i = trainer.test(model_path=args.model_path, data_provider=data_provider, test_size=test_size)


if __name__ == '__main__':
    main()
