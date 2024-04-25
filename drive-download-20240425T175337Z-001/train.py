## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
from customResNet152 import *
warnings.filterwarnings('ignore')

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_train.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        #scheduler.step()
        scheduler.step(loss)
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
        
    return [map.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist,mode="train")
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8)

## Loading the model to run
#model = timm.create_model('xception', num_classes=config.n_classes)
#model = timm.create_model('tf_efficientnet_b8', pretrained=True,num_classes=config.n_classes)
# Example usage
#model = CustomResNet152(num_classes=config.n_classes, pretrained=True, new_num_filters=128, dropout_rate=0.5, feature_extractor=False)
#model = CustomCNN(num_classes=config.n_classes)

#model = timm.create_model('resnet152', pretrained=True,num_classes=config.n_classes)
#print(model)
#print(model.num_features)
#model = timm.create_model('vit_large_patch16_224', pretrained=True,num_classes=config.n_classes)
#model = timm.create_model('rexnet_200', pretrained=True,num_classes=config.n_classes)
#model.fc = nn.Linear(model.num_features, config.n_classes)
#model = timm.create_model('cspdarknet53', pretrained=True,num_classes=config.n_classes)
model = timm.create_model('tf_efficientnetv2_l', pretrained=True,num_classes=config.n_classes)
#Batch normalization 
model.fc = nn.Sequential(
    nn.BatchNorm1d(model.num_features),
    nn.Linear(model.num_features, config.n_classes)
)

#Dropout layer
#model.fc = nn.Sequential(
#    nn.Dropout(0.5),
#    nn.Linear(model.num_features, config.n_classes)
#)

#model.fc = nn.Sequential(
#    nn.Linear(model.num_features, config.n_classes),
#    nn.ReLU()
#)


#model = timm.create_model('densenet201', pretrained=True,num_classes=config.n_classes)
#model = timm.create_model('convnexttiny', num_classes=config.n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
###Optimisers###
#optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.0005, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate, lr_decay=0, weight_decay=0.0005, initial_accumulator_value=0, eps=1e-10)
#optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
#optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate, rho=0.9, eps=1e-06, weight_decay=0.0005)

###Schedulers###
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
#scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
#scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)

criterion = nn.CrossEntropyLoss().cuda()
#criterion = nn.KLDivLoss()


############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
prevAccuracy = [0]
maxAccuracy = [0]
counter = 0
#train
for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)
    if(val_metrics > maxAccuracy):
      maxAccuracy = val_metrics
    if(prevAccuracy > val_metrics):
      counter+=1
      if(counter == 3):
        print("Current epoch", epoch)
        print("\nPrevious model was optimal")
        print("\n Previous Accuracy", prevAccuracy[0].item())
        print("\n Current Accuracy", val_metrics[0].item())
        print("\n Max Accuracy", maxAccuracy[0].item())
        break
    if epoch != 0:
      print("\n Previous Accuracy", prevAccuracy[0].item())
    prevAccuracy = val_metrics
    print("\n Current Accuracy", val_metrics[0].item())
    ## Saving the model
    filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
    torch.save(model.state_dict(), filename)
    

   
