import torch
import numpy
from torchvision.models import resnet as rn
from torchvision.transforms import transforms
from dataset_class import dataset_class as dc
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
import time
import copy
import sys
import os
import json
import shutil


cnt = 0

with open(os.path.join('./config.json'), 'r') as infile:
    config = json.load(infile)

# model_path = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/models/')
# model_folder = './models/{}'.format(config['version'])
model_folder = os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/models/', config['version'])
logs_folder = os.path.join(model_folder, 'logs')

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
    os.mkdir(logs_folder)
    print('created {}'.format(model_folder))
else:
    inp = input('Model set {} already exists. Continue?(y/yes)\t'.format(config['version']))
    if inp.lower() != 'y' and inp.lower() != 'yes':
        exit()


transform_pile = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

train_dataset = dc.ImageDataset('train', split=(0.8, 0.1), transform=transform_pile)
val_dataset = dc.ImageDataset('val', split=(0.8, 0.1), transform=transform_pile)
test_dataset = dc.ImageDataset('test', split=(0.8, 0.1), transform=transform_pile)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
print('cuda name: {}'.format(torch.cuda.get_device_name(1)))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
            running_corrects = 0

            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):
                batch_start = time.time()
                # print(len(sample[0]))
                # img = sample[0].to(device=device, dtype=torch.float)
                # scores = sample[1].to(device=device, dtype=torch.float)
                img = sample['img'].to(device=device, dtype=torch.float)
                scores = sample['scores'].to(device=device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, scores)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                batch_end = time.time()
                loss_data = loss.item()
                print(
                    '{} Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}\tTime: {:.3f}s/batch\t Total time: {:.0f}m {:.0f}s'.format(
                        phase,
                        epoch, i_batch * config['batch_size'], len(dataloaders[phase].dataset),
                        100. * i_batch * config['batch_size'] / len(dataloaders[phase].dataset), loss_data,
                        batch_end - batch_start, (time.time() - since) // 60, (time.time() - since) % 60
                        )
                )
                #running_corrects += torch.sum(outputs == scores)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            #epoch_acc = running_corrects.double() / len(dataloaders[phase])

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

dataloaders = {
    'train':    DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
    'val':  DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=10),
    }

for mod in config['models']:
    model_id = mod['id']
    print('Training model {}...'.format(model_id))
    print('Batch size: {}'.format(config['batch_size']))
    model_callback = eval(mod['model'])
    model = model_callback(pretrained=True, progress=True)
    if 'last_layer' in mod:
        model.fc = eval(mod['last_layer'].format(model.fc.in_features))
    if 'layers_to_unfreeze' in mod:
        layers = mod['layers_to_unfreeze']
        for name, param in model.named_parameters():
            if all([l not in name for l in layers]):
            # if not ('layer4' in name or 'fc' in name):
            #     print('{}\tFrozen'.format(name))
                param.requires_grad = False
            # else:
                # print('{}\tWarm'.format(name))
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
    model, lowest_loss = train_model(model, loss, optimizer, dataloaders, None, log_filename, num_epochs=25)
    torch.save(model.state_dict(), os.path.join(model_folder, mod['config'].format(lowest_loss)))
# model = rn.resnet101(pretrained=True, progress=True)
# # model.fc = Linear(512, 2)
# model.fc = Linear(2048, 1, bias=False)
# count = 0
# for name, param in model.named_parameters():
#     count += 1
#     if not ('layer4' in name or 'fc' in name):
#         print('{}\tFrozen'.format(name))
#         param.requires_grad = False
#     else:
#         print('{}\tWarm'.format(name))
#
# model = model.to(device)
# loss = MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# model, lowest_loss = train_model(model, loss, optimizer, dataloaders, None, num_epochs=25)
# torch.save(model.state_dict(), './models/model101(1 out)_unfrozen-full4_{}.config'.format(lowest_loss))
