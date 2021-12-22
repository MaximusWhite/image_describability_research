import torch
import numpy as np
from torchvision.models import resnet as rn
from torchvision.transforms import transforms
from torch.nn import functional as F
from dataset_class import dataset_class as dc
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss, L1Loss
import time
import copy
import sys
import os
import json
import shutil
import scipy
import matplotlib.pyplot as plt
import random
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

with open(os.path.join('./config.json'), 'r') as infile:
    config = json.load(infile)

model_folder = os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/models/CrissCrossOracle', config['version'])
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

transform_pile = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
gpu_id = 0 if len(sys.argv) == 1 else int(sys.argv[1])
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
print('cuda name: {}, ID: {}'.format(torch.cuda.get_device_name(gpu_id), gpu_id))
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, scheduler, log_filename, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = sys.float_info.max
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log_entry = ''
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Phase: ', phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):
                batch_start = time.time()
                # print(len(sample[0]))
                # img = sample[0].to(device=device, dtype=torch.float)
                # scores = sample[1].to(device=device, dtype=torch.float)
                img = sample['img'].to(device=device, dtype=torch.float)
                avg_cc = sample['avg_cc'].to(device=device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img)
                    #outputs = F.sigmoid(outputs)
                    loss = criterion(outputs, avg_cc)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()
                batch_end = time.time()
                loss_data = loss.item()
                if i_batch * config['batch_size'] % 10 == 0:
                    print(
                        '{} Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}\tTime: {:.3f}s/batch\t Total time: {:.0f}m {:.0f}s'.format(
                            phase,
                            epoch, 
                            i_batch * config['batch_size'], 
                            len(dataloaders[phase].dataset),
                            100. * i_batch * config['batch_size'] / len(dataloaders[phase].dataset), loss_data,
                            batch_end - batch_start, (time.time() - since) // 60, (time.time() - since) % 60
                            )
                    )

            epoch_loss = running_loss / len(dataloaders[phase])

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            log_entry += ('{},' if phase == 'train' else '{}').format(epoch_loss)  # t_loss, v_loss
        log_file = open(os.path.join(logs_folder, log_filename), 'a')
        log_file.write(log_entry+'\n')
        log_file.close()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(loss))

    # load best model weights.resnet101(pretrai
    model.load_state_dict(best_model_wts)
    return model, min_loss

def test_model(model, test_dataset, model_id):
    print('Testing {}...'.format(model_id))
    test_results = {
        'gt': [],
        'pred': [],
        'pearson_score': -9999
    }
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image = test_dataset[i]['img'].to(device=device, dtype=torch.float)
            gt = test_dataset[i]['avg_cc'].to(device=device, dtype=torch.float)
            pred = model(image[None])
            gt_num = gt.cpu().detach().numpy()
            pred_num = pred.cpu().detach().numpy()[0]
            predictions.append(pred_num)
            actuals.append(gt_num)
            test_results['gt'].append(gt_num)
            test_results['pred'].append(pred_num)
            if i % 100 == 0:
                print('\rProcessed {:4.4f}%'.format((i / len(test_dataset) * 100)), end="")
                sys.stdout.flush()
    print('\n')
    predictions = np.squeeze(np.array(predictions))
    actuals = np.squeeze(np.array(actuals))
    
    plt.clf()
    plt.scatter(predictions, actuals, c='blue', alpha=0.3)
    plt.title('{}'.format(model_id))
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.savefig(os.path.join(plots_folder, '{} RAW.png'.format(model_id)))
    plt.clf()
    plt.scatter(predictions, actuals, c='blue', alpha=0.3)
    plt.title('{}'.format(model_id))
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.Normalize(vmin=0.0, vmax=1.0)
    plt.savefig(os.path.join(plots_folder, '{} NORM.png'.format(model_id)))
    

    pearson_score = scipy.stats.pearsonr(actuals, predictions)[0]
    print('Pearson correlation coefficient: {}'.format(pearson_score))
    test_results['gt'] = (np.array(np.squeeze(test_results['gt'])).astype(str)).tolist()
    test_results['pred'] = (np.array(np.squeeze(test_results['pred'])).astype(str)).tolist()
    test_results['pearson_score'] = pearson_score
    return test_results
rand_search = False
if 'hp_random_search' in config and config['hp_random_search'] == True:
    rand_search = True
    fmt_targets = config['pre-format']['targets']
model_index = 0
for mod in config['models']:
    train_dataset = dc.ImageDataset('train', split=(0.8, 0.1), transform=transform_pile)
    val_dataset = dc.ImageDataset('val', split=(0.8, 0.1), transform=transform_pile)
    test_dataset = dc.ImageDataset('test', split=(0.8, 0.1), transform=transform_pile)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
        'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
    }

    if rand_search:
        lr_base = random.uniform(1,10)
        lr_exp = random.randrange(1, 10)
        lr = eval('{}e-{}'.format(lr_base, lr_exp))
        wd_base = random.uniform(1,10)
        wd_exp = random.randrange(1, 10)
        wd = eval('{}e-{}'.format(wd_base, wd_exp))
#         for tgt in fmt_targets:
#             config['models'][model_index][tgt] = mod[tgt].format(lr, wd, '{}')
        with open(os.path.join(model_folder, './config_backup_FORMATTED.json'), 'w') as config_backup:
            json.dump(config, config_backup, indent=2)

    model_id = mod['id']
    print('Training model {}...'.format(model_id))
    print('Batch size: {}'.format(config['batch_size']))
    
    ### MODEL INIT ###
    # ResNet from config
    model_callback = eval(mod['model'])
    model = model_callback(pretrained=True, progress=True)
        # AlexNet
#     model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        # VGG
#     model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
    if 'last_layer' in mod:
#         model.classifier[6] = eval(mod['last_layer'].format(model.classifier[6].in_features, len(targets)))
        model.fc = eval(mod['last_layer'].format(model.fc.in_features, 1))
    if 'layers_to_unfreeze' in mod:
        layers = mod['layers_to_unfreeze']
        if len(layers) > 0:
            for name, param in model.named_parameters():
    #             print(name)
                if all([l not in name for l in layers]):
                    param.requires_grad = False
    if 'loss' in mod:
        loss = (eval(mod['loss']))()
    else:
        loss = MSELoss()
    if 'optimizer' in mod:
        optimizer = eval(mod['optimizer'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.to(device)
    log_filename = '{}-LOG.log'.format(model_id)
    model_index += 1
    model, lowest_loss = train_model(model, loss, optimizer, dataloaders, None, log_filename, num_epochs=25)
    torch.save(model.state_dict(), os.path.join(model_folder, mod['config'].format(lowest_loss)))
#     model.load_state_dict(torch.load(os.path.join(model_folder, 'v1-cc_oracle model18_unfrozen-full (lr=1e-6,wd=1e-8)_0.4372740912437439.config')))
    # TEST
    results = test_model(model, test_dataset, model_id)
    with open(os.path.join(results_folder, mod['test_result_filename']), 'w') as outfile:
        json.dump(results, outfile)