from args import get_args
from trainer import GMTrainer
import torch
import os
import math
import sys
from datetime import datetime
from logger import setup_logging

if __name__ == '__main__':
    log_file = setup_logging()
    start_time = datetime.now()
    print(f"train start time: {start_time}")
    minloss = math.inf
    minloss_val = math.inf
    minloss_val_epoch = 0
    args = get_args()
    print(f"training args: {args}")
    testrun = GMTrainer(args)
    epochs = args.epoch
    directory = 'models'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dic1 = os.path.join(directory, timestamp)
    subdirectory1 = os.path.join(dic1, 'saved_models')
    subdirectory2 = os.path.join(dic1, 'best_models')
    test_result_pic = args.result_pic_path
    if not os.path.exists(dic1):
        os.makedirs(dic1)
        os.makedirs(subdirectory1)
        os.makedirs(subdirectory2)
    if not os.path.exists(test_result_pic):
        os.makedirs(test_result_pic)
    for epoch in range(epochs):
        loss, grad_norm,scale_log2 =testrun.train()
        loss = loss.item()
        
        print(f'epoch {epoch} loss is {loss}')
       
        ckpt = os.path.join(subdirectory2, 'best_model.pth')
        ckpt2 =  'model' + str(epoch) + '.pth'
        ckpt3 = os.path.join(subdirectory1, ckpt2)
        val_loss = testrun.val()
        if val_loss > 0 and val_loss < minloss_val:
            minloss_val = val_loss
            if args.name != 'electricity' :
                torch.save(testrun.state_dict(), ckpt3)
            minloss_val_epoch = epoch
            torch.save(testrun.state_dict(), ckpt)
        print (f'min val_loss is {minloss_val} in epoch {minloss_val_epoch}')
        print (f'val_loss is {val_loss} in epoch {epoch}')
        if epoch % 5 == 0 :
            torch.save(testrun.state_dict(), ckpt3)
        if loss > 0 and loss < minloss:
            minloss = loss
            print (f'min loss is {minloss} in epoch {epoch}')
            
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"train end time: {end_time}")
    print(f"total train period: {training_time}")
    testrun.load_state_dict(torch.load(ckpt))
    
    testloss_mae,testloss_mse,testloss_mre = testrun.test()
    print(f"test dataset mae loss: {testloss_mae}")
    print(f"test dataset mse loss: {testloss_mse}")
    print(f"test dataset mre loss: {testloss_mre}")
    # testloss_mae,testloss_mse,testloss_mre = testrun.test_onepiece_05()
    # print(f"test dataset mae loss: {testloss_mae}")
    # print(f"test dataset mse loss: {testloss_mse}")
    # print(f"test dataset mre loss: {testloss_mre}")
    # testloss_mae,testloss_mse,testloss_mre = testrun.test_onepiece_01()
    # print(f"test dataset mae loss: {testloss_mae}")
    # print(f"test dataset mse loss: {testloss_mse}")
    # print(f"test dataset mre loss: {testloss_mre}")
