from Common import *
from utils.vis_tool import Visualizer
from config import opt
from utils.loss import DiceLossPlusCrossEntrophy
from Generate_Random_Samples_For_Training import Generate_Random_Samples
from utils.meters import AverageMeter
import torch
import time
from utils.eval_metric import ConfusionMeter
from Validation import val_model
from utils.tools import *
import torch.nn as nn

all_mean1 = np.array([168.5], dtype=np.float32)
all_std1 = np.array([500], dtype=np.float32)

##### used for training:
def TrainOneEpoch(train_loader,model,optimizer,criterion,epoch_num,vis_tool,prefix):
    losses = AverageMeter()
    train_eval = ConfusionMeter(num_class=opt.out_dim)

    # calculate the final result
    train_dice=AverageMeter()
    train_recall=AverageMeter()

    model.train()

    for batch_ids, (data, target) in enumerate(train_loader):
        if opt.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        # calculate the weight of the batch:
        weight = GetWeight(opt,target, slr=0.00001,is_t=0)
        loss = criterion(output, target,weight=weight)

        loss.backward()
        optimizer.step()

        # update the loss value
        losses.update(loss.item())

        # calculate the metrics for evaluation:
        _, pred = torch.max(output, 1)
        train_eval.update(pred, target)

        avg_loss=losses.avg
        dice_value=train_eval.get_scores('Dice')
        recall_value=train_eval.get_scores('Recall')

        train_dice.update(dice_value)
        train_recall.update(recall_value)

        # begin to show the results
        if batch_ids % opt.train_plotfreq == 0:
            vis_tool.plot('Train_Loss', loss.item())
            vis_tool.plot('Train_Dice', dice_value)
            vis_tool.plot('Train_Recall', recall_value)

            # begin to plot the prediction result
            image1=data.cpu().numpy()[0,0,...]
            image1 = image1 * all_std1 + all_mean1
            image1=np.clip(image1,150,350)
            image1 = (image1 - 150) /200
            image1_mip=np.hstack([np.max(image1,0),np.max(image1,1),np.max(image1,2)])

            # see the pred
            pred1=pred.cpu().numpy()
            pred1=pred1[0,...]
            mip1=np.hstack([np.max(pred1,0),np.max(pred1,1),np.max(pred1,2)])

            # see the label
            target1=target.cpu().numpy()
            target1 =target1[0,...]
            mip2 = np.hstack([np.max(target1, 0), np.max(target1, 1), np.max(target1, 2)])
            mip3=np.vstack([image1_mip,mip1,mip2])
            vis_tool.img('pred_label',np.uint8(255*mip3))


        print('Train:Batch_Num:{}  Loss:{:.3f}  Dice:{:.3f}  Recall:{:.3f}'.format(batch_ids,loss.item(),dice_value,recall_value))


    return avg_loss,train_dice.avg,train_recall.avg


# define the main function
def main():
    ##########################################################
    # 1. generate random cropped samples from the selected images for training
    if opt.random_sample_flag==1:
        Generate_Random_Samples(opt)

    # 2. load the paramter and print them
    opt._parse()

    # load the dataloader
    train_loader = GetDatasetLoader(opt, phase='train')
    val_loader=GetDatasetLoader(opt,phase= 'val')

    # get the net work
    model=GetModel(opt)

    # if opt.load_state:
    #     ##### fine tune the result
    #     parameters_name = 'checkpoints/'+opt.dataset_prefix+'_epoch_100.ckpt'
    #     model_CKPT = torch.load(parameters_name)
    #     model.load_state_dict(model_CKPT['state_dict'])

    optimizer=GetOptimizer(opt,model)
    scheduler=GetScheduler(opt,optimizer)

    # get the loss function
    criterion=DiceLossPlusCrossEntrophy()

    vis_tool=Visualizer(env='VoxResNet_3D')
    log_path = 'result'
    log_name = os.path.join(log_path, 'log_{}.txt'.format(opt.dataset_prefix))
    log=WriteLog(log_path,log_name,opt)
    log.write('selected dataset \n')
    log.write('weighted and dice: 0.5:1 \n')
    log.write('epoch |train_loss |train_dice |train_recall |valid_loss |valid dice |valid_recall |time          \n')
    log.write('------------------------------------------------------------\n')


    # begin to train:
    for epoch_num in range(opt.train_epoch):
    # for epoch_num in np.arange(1,30):
        start_time=time.time()
        scheduler.step()

        avg_loss,train_dice,train_recall=\
            TrainOneEpoch(train_loader, model, optimizer, criterion, epoch_num, vis_tool,opt.dataset_prefix)

        # the information
        run_time = time.time()-start_time
        print('Train Epoch{} run time is:{:.3f}m and {:.3f}s'.format(epoch_num,run_time//60,run_time%60))
        print('Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(avg_loss,train_recall,train_dice))


        if epoch_num>=0:
            save_name1 = (opt.dataset_prefix+'_epoch_{}').format(epoch_num)
            state={'epoch': epoch_num + 1,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   }

            save_parameters(state, save_name1)

        # judge whether to test or not
        if opt.val_run:
            val_avgloss,val_dice,val_recall=val_model(opt,val_loader,model,criterion,vis_tool)
            print('Test Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(val_avgloss,val_recall,val_dice))

        log.write('%d |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f \n' % (epoch_num,avg_loss,train_dice,train_recall,
                                                                    val_avgloss,val_dice,val_recall,run_time))
        log.write('\n')




if __name__=="__main__":
    main()



















































