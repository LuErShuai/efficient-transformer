# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from collections import namedtuple
import time
import random 

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    # model.agent.reward_one_epoch = 0
    
    # torch.cuda.empty_cache()
    sample_num = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        start = time.perf_counter()
        sample_num += 1

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        

        # size of loss:[batch_size]
        loss_value = loss.item()

        if args.train_agent:
            torch.cuda.empty_cache()
            end_1 = time.perf_counter()
            # train agent
            # classify_results = outputs - targets
            _, outputs_max_index = outputs.max(dim=1)
            _, targets_max_index = targets.max(dim=1)
            # self.buffer = 
            # {
            #     "state":[], -> [block_num, batch_size, token_num, token_dim]
            #     "state_next":[], 
            #     "action":[],
            #     "action_prob":[]
            # }
            Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
            buffers = model.buffer

            batch_size = buffers["state"][0].shape[0]
            token_num  = buffers["state"][0].shape[1]
            block_num  = len(buffers["state"])
            token_keep_ratio = buffers["token_keep_ratio"][0]
            # token_keep_ratio = 0

            time_2 = time.perf_counter()

            # for i in range(batch_size):
            for i in range(1):
            # for i in random.sample(range(1,batch_size), 5):
                if outputs_max_index[i] == targets_max_index[i]:
                    classify_correct = True 
                else:
                    classify_correct = False

                # for j in range(token_num):
                for j in random.sample(range(1,token_num), 15):
                    del model.agent.buffer[:] # clear experience
                    token_done = False
                    for k in range(block_num):
                        if token_done:
                            break
                        # size of buffers["state"]: [block_num, batch_size, token_num, token_dim]
                        state = buffers["state"][k][i][j]
                        action = buffers["action"][k][i][j]
                        if action == 0:
                            token_done = True
                        action_prob = buffers["action_prob"][k][i][j]
                        state_next = buffers["state_next"][k][i][j]

                        reward = caculate_reward_per_step(k, classify_correct,
                                                           action, token_keep_ratio,
                                                          model.agent.total_steps)
                        trans = Transition(state.detach().cpu().numpy(), action, action_prob,
                                           reward, state_next.detach().cpu().numpy())
                        model.agent.store_transition(trans)
                        model.agent.total_steps += 1

                        
                    
                    # if len(model.agent.buffer) > model.agent.batch_size:
                    #     model.agent.update()
                if len(model.agent.buffer) > 0:
                    model.agent.update()

            time_3 = time.perf_counter()

            # if utils.is_main_process() and model.agent.training_step > 50000:
            if sample_num%100 == 0:
                model.agent.save_param()
                print(model.agent.total_steps)
                print("-------------------save ppo weight-------------------")
                # return

            end_2 = time.perf_counter()
            run_time_deit = end_1 -start
            run_time_agent = end_2 - end_1 
            # print("run time deit:", run_time_deit * 1000)
            # print("run time cycle", (time_3 - time_2) * 1000)
            # print("run time agent:", run_time_agent * 1000)





            # for j in range(block_num):
            #     # size of buffer["state"]: [block_num, batch_size, token_num, token_dim]
            #     # size of state: [token_num, token_dim]
            #     state = buffers["state"][j][i]
            #     action = buffers["action"][j][i]
            #     action_prob = buffers["action_prob"][j][i]
            #     state_next  = buffers["state_next"][j][i]
            #     # reward = caculate_reward(j, classify_correct, action)

            #     token_num = state.shape[0]
            #     for k in range(token_num):
            #         reward = caculate_reward_per_token(j, classify_correct,
            #                                            action[k])
            #         trans = Transition(state[k], action[k], action_prob[k],
            #                            reward, next_state[k])
            #         model.agent.store_transition(trans)

            #         # one image with 12 block
            #         # which means 12 step/transition for rl
            #         # after 12 transition store into agent.buffer
            #         # mean this trajecotry for rl is done
            #         # so the update should being excuted
            #         if len(model.agent.buffer) > model.agent.batch_size:
            #             model.agent.update()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # if args.train_deit or args.fine_tune:
        #     optimizer.zero_grad()

        #     # this attribute is added by timm on one optimizer (adahessian)
        #     is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        #     loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        # optimizer.zero_grad()

        # # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['reward_batch'].update(reward_one_batch, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def caculate_reward_per_step(num_block, classify_correct, action, token_keep_ratio,
                             total_steps):
    reward_for_classify = 1
    reward_for_action = 2

    # simplest: split equally
    if classify_correct:
        reward_1 = 1.0*reward_for_classify

        # reward_2 = (1 - action)*2.5*(12-num_block)
        # reward_3 = 0
        # reward_3 = -action*num_block*0.125
        # reward_1 = reward_for_classify/12
        # reward_2 = reward_for_action
        
        # reward_3 = (1 - action)*0.1
    else:
        # reward_1 = -reward_for_classify/12
        # reward_2 = - reward_for_action
        # reward_1 = 0
        reward_1 = -1*reward_for_classify
        # reward_2 = 0
        # reward_3 = 0
        # reward_3 = -(1 - action)*0.1

    # reward_2 = (1 - action)*16*(12-num_block)
    if classify_correct:
        # reward_2 = (1 - action)
        if action == 1:
            reward_2 = 0
        if action == 0:
            reward_2 = 1.0
    else:
        if action == 1:
            # reward_2 = -0.5
            reward_2 = 0
        if action == 0:
            reward_2 = 0
    # reward_2 = (1 - action)*100*(12-num_block)
    # reward_2 = 1 - action
    # reward_3 = -action*num_block*0.0125
    reward_3 = 0

    
    eta = 32
    # reward_4 = - math.exp(eta*abs(token_keep_ratio - 0.7))
    
    # d = token_keep_ratio - 0.75
    # if d > 0:
    #     reward_4 = - action*math.exp(eta*abs(d))
    # if d <= 0:
    #     reward_4 = - (1-action)*math.exp(eta*abs(d))

    reward_2 = 25*(math.exp((1-token_keep_ratio)) - 1) - 4*math.exp(token_keep_ratio)
    # reward_2 = 0
    
    # if token_keep_ratio > 0.75:
    #     reward_4 = -2*action*math.exp(eta*abs(token_keep_ratio-0.75))
    # elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.55:
    #     reward_4 = 2 - math.exp(10*abs(token_keep_ratio - 0.65))
    # elif token_keep_ratio <= 0.55:
    #     reward_4 = -2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.55))

    if total_steps < 30000:
        if token_keep_ratio > 0.75:
            reward_4 = -2-2*action*math.exp(eta*abs(token_keep_ratio-0.75))
        elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.70:
            reward_4 = -2*action*math.exp(eta*abs(token_keep_ratio-0.7))
        elif token_keep_ratio <= 0.70 and token_keep_ratio > 0.60:
            reward_4 = (1-token_keep_ratio)*1.5
        elif token_keep_ratio <= 0.60 and token_keep_ratio > 0.55:
            reward_4 = -2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.6))
        elif token_keep_ratio <= 0.55:
            reward_4 = -2-2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.55))
    else:
        # if token_keep_ratio > 0.75:
        #     reward_4 = -8
        # elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.70:
        #     reward_4 = -4
        # elif token_keep_ratio <= 0.70 and token_keep_ratio > 0.60:
        #     reward_4 = 4 - 2*math.exp(10*abs(token_keep_ratio - 0.65))
        # elif token_keep_ratio <= 0.60 and token_keep_ratio > 0.55:
        #     reward_4 = -4
        # elif token_keep_ratio <= 0.55:
        #     reward_4 = -8
        
        # if token_keep_ratio - 0.65 > 0:
        #     reward_4 = -action*4*(math.exp(token_keep_ratio - 0.65)-1)
        # else:
        #     reward_4 = -(1-action)*4*(math.exp(abs(token_keep_ratio - 0.65))-1)

        # if token_keep_ratio > 0.8:
        #     reward_4 = -action*1*(math.exp(eta*(token_keep_ratio - 0.70))-1)
        # elif token_keep_ratio < 0.6:
        #     reward_4 = -(1-action)*1*(math.exp(eta*abs(token_keep_ratio - 0.70))-1) 
        # else:
        #     # reward_4 = -1*(math.exp((eta)*abs(token_keep_ratio - 0.70))) 
        #     reward_4 = 25*(math.exp((1-token_keep_ratio)) - 1) - 4*math.exp(token_keep_ratio)

        reward_4 = - (math.exp(abs(token_keep_ratio - 0.70))-1)
        # reward_4 = 0


    eta = 0.6
    beta = 0.80
    return eta*reward_1 + (1-eta)*(beta*reward_2 + (1-beta)*reward_4)
    # return reward_2
    # return reward_1 + reward_2

def caculate_reward(num_block, classify_correct, action):
    # size of action: [token_num] -> 197
    # action for 197 tokens in one image

    reward_for_classify = 24 
    # simplest: split equally
    if classify_correct:
        reward_1 = reward_for_classify/12
    else:
        reward_1 = -reward_for_classify/12

    reward_for_action = 1
    reward = torch.empty(action.shape, device=action.device)
    for i in range(len(action)):
        # action: 0:discard token 
        #         1:keep token
        reward_2 = 0
        reward_2 += (1 - action[i])*reward_for_action

        reward_total = reward_1 + reward_2
        reward[i] = reward_total 
        
    return reward

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
