import torch
import numpy as np
from torchvision.models import resnet as rn
from torchvision.transforms import transforms
from torch.nn import functional as F
from embedding_dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss
import torch.nn as nn
import time
import copy
import sys
import os
import json
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
import random
from confidence_network import SemaConfNet as semaconfnet
from confidence_network import SemaConfNetSplit as semaconfnetsplit


with open(os.path.join('./config.json'), 'r') as infile:
    config = json.load(infile)

model_folder = os.path.join('../../../../models/SemaConfNet/', config['version'])
logs_folder = os.path.join(model_folder, 'logs')
results_folder = os.path.join(model_folder, 'results')
plots_folder = os.path.join(model_folder, 'plots')

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
    os.mkdir(logs_folder)
    os.mkdir(results_folder)
    os.mkdir(plots_folder)
    print('created {}'.format(model_folder))
else:
    inp = input('Model set {} already exists. Continue?(y/yes)\t'.format(config['version']))
    if inp.lower() != 'y' and inp.lower() != 'yes':
        exit()
        
shutil.copyfile(os.path.join('./config.json'), os.path.join(model_folder, './config_backup.json'))

gpu_id = 0 if len(sys.argv) == 1 else int(sys.argv[1])
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
print('cuda name: {}, ID: {}'.format(torch.cuda.get_device_name(gpu_id), gpu_id))
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, scheduler, log_filename, num_epochs=25):
    since = time.time()
    best_acc_model_wts = copy.deepcopy(model.state_dict())
    best_loss_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = sys.float_info.max
    max_acc = sys.float_info.min
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log_entry = ''
        acc_log_entry = ''
        mon_log_entry = ''
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Phase: ', phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            gts = np.array([])
            predictions = np.array([])
            
            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):
                batch_start = time.time()
                
                embed = sample['data'].to(device=device, dtype=torch.float)
                true_pair = sample['is_true_pair'].to(device=device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(embed)
#                     if (i_batch == 3 and phase == 'val'):
#                         print(outputs.cpu().detach().numpy().flatten())
#                         test_out = outputs.cpu().detach().numpy().flatten()
                        
# #                         print('diff: {}'.format(0.5 - test_out))
#                         borderline_threshold = np.count_nonzero(abs(0.5 - test_out) < 0.1)
#                         print('BORDERLINE THRESHOLD: {}/{} ({}%)'.format(borderline_threshold, len(test_out), (borderline_threshold/len(test_out)) * 100))
                    loss = criterion(outputs, true_pair)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                
                batch_gts = torch.round(true_pair).cpu().detach().numpy().flatten()
                batch_preds = torch.round(outputs).cpu().detach().numpy().flatten()
                
#                 gt_true_percentage = np.count_nonzero(batch_gts) / len(batch_gts)
#                 prediction_true_percentage = np.count_nonzero(batch_preds) / len(batch_preds)
                
#                 print("GT 1's: {}".format(np.count_nonzero(batch_gts) / len(batch_gts)))
#                 print("Pred 1's: {}".format(np.count_nonzero(batch_preds) / len(batch_preds)))
                
                gts = np.append(gts, batch_gts)
                predictions = np.append(predictions, batch_preds)
                
                batch_gts[batch_gts <= 0.5] = 0.
                batch_gts[batch_gts > 0.5] = 1.
                
                batch_preds[batch_preds <= 0.5] = 0.
                batch_preds[batch_preds > 0.5] = 1.
                
                running_acc = accuracy_score(batch_gts, batch_preds)
                running_loss += loss.item()
                batch_end = time.time()
                loss_data = loss.item()
                if i_batch * config['batch_size'] % 800 == 0:
                    print(
                        '{} Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}\tAcc: {:.2f}\tTime: {:.3f}s/batch\t Total time: {:.0f}m {:.0f}s'.format(
                            phase,
                            epoch, 
                            i_batch * config['batch_size'], 
                            len(dataloaders[phase].dataset),
                            100. * i_batch * config['batch_size'] / len(dataloaders[phase].dataset), 
                            loss_data,
                            running_acc,
                            batch_end - batch_start, (time.time() - since) // 60, (time.time() - since) % 60
                            )
                    )
                    
            gts[gts <= 0.5] = 0.
            gts[gts > 0.5] = 1.

            predictions[predictions <= 0.5] = 0.
            predictions[predictions > 0.5] = 1.        
            epoch_loss = running_loss / len(dataloaders[phase])
            
#             print('EPOCH GTs: {}'.format(gts))
#             print('EPOCH PREDs: {}'.format(predictions))
            epoch_acc = accuracy_score(gts, predictions)
            print('{} Loss: {:.4f}; Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < min_loss and (np.count_nonzero(predictions) / len(predictions)) > 0.0 and (np.count_nonzero(predictions) / len(predictions)) < 1.0:
                    min_loss = epoch_loss
                    best_loss_model_wts = copy.deepcopy(model.state_dict())
                if epoch_acc > max_acc:
                    max_acc = epoch_acc
                    best_acc_model_wts = copy.deepcopy(model.state_dict())
                    
            log_entry += ('{},' if phase == 'train' else '{}').format(epoch_loss)  # t_loss, v_loss
            acc_log_entry += ('{},' if phase == 'train' else '{}').format(epoch_acc)
            mon_log_entry += ('{},{},' if phase == 'train' else '{},{}').format(np.count_nonzero(gts) / len(gts), np.count_nonzero(predictions) / len(predictions))
        
        mon_log_file = open(os.path.join(logs_folder, 'mon_' + log_filename), 'a')
        mon_log_file.write(mon_log_entry+'\n')
        mon_log_file.close()
        
        acc_log_file = open(os.path.join(logs_folder, 'acc_' + log_filename), 'a')
        acc_log_file.write(acc_log_entry+'\n')
        acc_log_file.close()
        
        log_file = open(os.path.join(logs_folder, log_filename), 'a')
        log_file.write(log_entry+'\n')
        log_file.close()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(min_loss))
    print('Best val accuracy: {:3f}'.format(max_acc))

    # load best model weights
#     model.load_state_dict(best_model_wts)
    
    return best_loss_model_wts, best_acc_model_wts, min_loss, max_acc


def test_model(model, test_dataset, model_id, mode='Lowest Loss'):
    print('Testing {}...'.format(model_id))
    result = {}
    model.eval()
    predictions = []
    actuals = []
    
    preds = []
#     metas = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            embed = torch.from_numpy(np.array([sample['data']])).to(device=device, dtype=torch.float)
            gt = sample['is_true_pair'].to(device=device, dtype=torch.float)
            pred = model(embed)
            meta = sample['meta']
#             meta = test_dataset[i]['meta']
#             if meta in metas:
#                 print('REPEAT')
#             metas.append(meta)
#             print('{}'.format(gt.shape))
            preds.append([meta, gt.cpu().detach().numpy()[0].item(), pred.cpu().detach().numpy()[0].item()])
            actuals.append(gt.cpu().detach().numpy()[0].item())
            predictions.append(pred.cpu().detach().numpy()[0].item())
#             print('true: {}; pred: {}/{}'.format(gt.cpu().detach().numpy()[0].item(), torch.round(pred).cpu().detach().numpy()[0].item(), pred.cpu().detach().numpy()[0].item()))
    
    result['pred'] = list(predictions)
    result['gt'] = list(actuals)
    
    actuals = np.array(actuals)        
    actuals[actuals <= 0.5] = 0.
    actuals[actuals > 0.5] = 1.
    
    predictions = np.array(predictions) 
    predictions[predictions <= 0.5] = 0.
    predictions[predictions > 0.5] = 1.   
    result['acc'] = accuracy_score(actuals, predictions).item()
#     print('len test: {}'.format(len(test_dataset)))
    
    print('Mode: {}'.format(mode))
    print('Test accuracy: {}'.format(accuracy_score(actuals, predictions).item()))
    print('(% of predicted 1s: {})'.format(np.count_nonzero(predictions) / len(predictions)))
    print('(% of GT 1s: {})'.format(np.count_nonzero(actuals) / len(actuals)))
    
    with open('../../../../datasets/semaconf_preds.json', 'w') as f:
        json.dump(preds, f)
#     np.save(open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/DD_test.npy', "wb"), test_dataset.get_samples())
    return result


for mod in config['models']:
    if 'precomputed_dataset' in config:
        precomputed_dataset = config['precomputed_dataset']
#     train_dataset = EmbeddingDataset('train', precomputed_dataset, split=(0.7, 0.1))
#     val_dataset = EmbeddingDataset('val', precomputed_dataset, split=(0.7, 0.1))
    test_dataset = EmbeddingDataset('test', precomputed_dataset, split=(0.7, 0.1))
    
#     dataloaders = {
#         'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
#         'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
#     }
    
    model_id = mod['id']
    print('Training model {}...'.format(model_id))
    print('Batch size: {}'.format(config['batch_size']))
    
    model_callback = eval(mod['model'])
    model = model_callback()
    model = model.to(device)
    
    if 'epochs' in mod:
        epochs = mod['epochs']
    else:
        epochs = 100
    
    if 'loss' in mod:
        loss = (eval(mod['loss']))()
    else:
        loss = MSELoss()
    if 'optimizer' in mod:
        optimizer = eval(mod['optimizer'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    log_filename = '{}-LOG.log'.format(model_id)
    
#     loss_wts, acc_wts, lowest_loss, max_acc = train_model(model, loss, optimizer, dataloaders, None, log_filename, num_epochs=epochs)
    
    ## main way
    """ 
    model.load_state_dict(loss_wts)
    torch.save(model.state_dict(), os.path.join(model_folder, mod['config'].format('LOSS', lowest_loss)))
    results = test_model(model, test_dataset, model_id, 'Lowest Loss') 
       
    with open(os.path.join(results_folder, mod['test_result_filename'].format('LOSS')), 'w') as outfile:
        json.dump(results, outfile)
        
    model.load_state_dict(acc_wts)
    torch.save(model.state_dict(), os.path.join(model_folder, mod['config'].format('ACC', max_acc)))
    results = test_model(model, test_dataset, model_id, 'Highest Accuracy') 
    
    with open(os.path.join(results_folder, mod['test_result_filename'].format('ACC')), 'w') as outfile:
        json.dump(results, outfile)
    """

#     results = test_model(model, test_dataset, model_id, 'BASELINE') 
    
#     with open(os.path.join(results_folder, mod['test_result_filename'].format('BASELINE')), 'w') as outfile:
#         json.dump(results, outfile)

    model.load_state_dict(torch.load(os.path.join('../../../../models/SemaConfNet/v10-resnet_embeds-MP3-4/v10-resnet_embeds-MP3-4 SemaConfNetSplit (lr=1e-5,wd=1e-8)_ACC-0.9586856728678425.config'), map_location=device))
    results = test_model(model, test_dataset, 'v10-resnet_embeds-MP3', 'Highest Accuracy') 
    
