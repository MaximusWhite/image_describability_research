import torch
import numpy
from resnet_implementation import resnet as rn
from torchvision.transforms import transforms
from dataset_class import dataset_class as dc
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
import time
import copy
import sys
import os
import json


cnt = 0

def my_collate(batch):
    global cnt
    cnt += 1
    print('Batch {}'.format(cnt))
    max_1size = batch[0]['img'].shape[1]
    max_2size = batch[0]['img'].shape[2]
    for item in batch:
        # print('img type = {}, size = {}'.format(type(item['img']), item['img'].shape))
        max_1size = max_1size if item['img'].shape[1] < max_1size else item['img'].shape[1]
        max_2size = max_2size if item['img'].shape[2] < max_2size else item['img'].shape[2]

    print('max 1 = {}, max 2 = {}'.format(max_1size, max_2size))

    for item in batch:
        item['img'] = (item['img'].type(torch.FloatTensor)).unsqueeze(0)    # reshape to get 4th dimension (for pad)
        print(item['img'].size())
        pad_2d = max_1size - item['img'].shape[2]
        pad_3d = max_2size - item['img'].shape[3]
        if pad_3d > item['img'].shape[3]:
            #TODO: intermediate padding 3d
            pass
        if pad_2d > item['img'].shape[2]:
            #TODO: intermediate padding 2d
            pass

        # print('diff 3 = {}'.format())
        # print('diff 2 = {}'.format()
        print('pad tuple = {}'.format((0, max_2size - item['img'].shape[3], 0, max_1size - item['img'].shape[2])))
        item['img'] = torch.nn.functional.pad(item['img'], (0, max_2size - item['img'].shape[3], 0, max_1size - item['img'].shape[2]), 'reflect')
        # item['img'] = torch.nn.ReflectionPad1d((0, int(max_size - item['img'].shape[1])))
        # print('2 ::: img type = {}, size = {}'.format(type(item['img']), item['img'].shape))
        item['img'] = torch.squeeze(item['img'], 0)
    data = [item['img'] for item in batch]
    target = [item['scores'] for item in batch]
    target = torch.FloatTensor(target)
    data = torch.stack(data)
    return data, target

with open(os.path.join('./config.json'), 'r') as infile:
    config = json.load(infile)

model_folder = './models/{}'.format(config['version'])

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
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
print('cuda name: {}'.format(torch.cuda.get_device_name(0)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = sys.float_info.max
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

                if i_batch % 10 == 0:
                    batch_end = time.time()
                    loss_data = loss.item()
                    print(
                        '{} Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}\tTime: {:.3f}s/batch\t Total time: {:.0f}m {:.0f}s'.format(phase,
                            epoch, i_batch, len(dataloaders[phase].dataset),
                            100. * i_batch / len(dataloaders[phase].dataset), loss_data, batch_end - batch_start, (time.time() - since) // 60, (time.time() - since) % 60
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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(loss))

    # load best model weights.resnet101(pretrai
    model.load_state_dict(best_model_wts)
    return model, min_loss

dataloaders = {
    'train':    DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10),
    'val':  DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=10),
    }

for mod in config['models']:
    model_id = mod['id']
    print('Training model {}...'.format(model_id))
    model_callback = eval(mod['model'])
    model = model_callback(pretrained=True, progress=True)
    if 'last_layer' in mod:
        model.fc = eval(mod['last_layer'].format(model.fc.in_features))
    if 'layers_to_freeze' in mod:
        layers = mod['layers_to_freeze']
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
    model, lowest_loss = train_model(model, loss, optimizer, dataloaders, None, num_epochs=25)
    torch.save(model.state_dict(), mod['config'].format(lowest_loss))
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
