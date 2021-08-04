import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from fun_utils import *

def trainDFA(model, num_epochs):
    torch.manual_seed(6926)
    
    transform = get_basic_transformation()

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)

    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = False, transform = transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 0)
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Loss function
    criterion = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        
    print("\n\n=== Starting model training with %d epochs:\n" % (num_epochs,))        
    for epoch in range(1, num_epochs + 1):
        # Training
        print("EPOCH NUMBER " + str(epoch))
        train_epoch(model, train_loader, optimizer, criterion)


def train_epoch(model, train_loader, optimizer, criterion):
    
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        
        # Create tensor with dim 4 (batch size), 10 (no. of classes) with zeros everywhere,
        # except in the correct class, where it is 1.
        targets = torch.zeros(4, 10).scatter_(1, label.unsqueeze(1), 1.0)
        
        optimizer.zero_grad()
        output = model(data, targets)
        print("----")
        #print("data is " + str(data))
        #print("data size is "+ str(data.size()))
        #print("targets is " + str(targets))
        print("output is " + str(output))
        loss_val = criterion[0](output, criterion[1](targets))
        print("loss is:" + str(loss_val))
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()


def test_epoch(model, test_loader, loss, phase):
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)
    
    with torch.no_grad():
        for data, label in test_loader:
            targets = torch.zeros(label.shape[0], 10)
            output = model(data, None)
            test_loss += loss[0](output, loss[1](targets)).item()
            pred = output.max(1, keepdim=True)[1]
            
            correct += pred.eq(label.view_as(pred)).sum().item()
    print("Test loss is:")
    print(str(test_loss))
    print("len_dataset is:")
    print(str(len_dataset))
    loss = test_loss / len_dataset
    acc = 100. * correct / len_dataset
    print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))