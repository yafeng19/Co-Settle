import math
import sys
import torch
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    print("@@@@@@@@@@@@@@@ Training Parameters @@@@@@@@@@@@@@@")
    print("use_adapter: ", args.use_adapter)
    print("use_asy_pos_embedding: ", args.use_asy_pos_embedding)
    print("temp: ", args.temp)
    print("asy_pos_ratio: ", args.asy_pos_ratio)
    print("lambda_reg: ", args.lambda_reg)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    

    for data_iter_step, samples_tuple in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
       
        imgs_lst = [x.to(device, non_blocking=True) for x in samples_tuple]

        with torch.cuda.amp.autocast():
            if args.base_model == 'clip':
                forward_func = model.module.encode_image
            else:
                forward_func = model

            loss, acc, sim_identity = forward_func(imgs_lst, use_adapter=args.use_adapter, 
                                            use_asy_pos_embedding=args.use_asy_pos_embedding, temp=args.temp, 
                                            asy_pos_ratio=args.asy_pos_ratio, lambda_reg=args.lambda_reg, is_training=True)
            

        loss_value = loss.item()
        acc_value = acc.item()
        sim_identity_value = sim_identity.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc_value)
        metric_logger.update(sim_identity=sim_identity_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        acc_value_reduce = misc.all_reduce_mean(acc_value)
        sim_identity_reduce = misc.all_reduce_mean(sim_identity_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('cycle_acc', acc_value_reduce, epoch_1000x)
            log_writer.add_scalar('sim_identity', sim_identity_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}