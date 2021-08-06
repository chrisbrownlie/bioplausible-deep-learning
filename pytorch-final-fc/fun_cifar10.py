from fun_utils import get_basic_transformation
import torch
import torchvision
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def train_CIFAR10(model, model_name, spiking=False, debug=False):
    """
    Function to train model on CIFAR-10 Image Classification

    Args:
    model -- the class of the model to train
    model_name -- a unique name for the model, used for saving to models/ folder
    """
    torch.manual_seed(42)
    if Path("./models/" + model_name + ".pth").exists():
        print("Trained model already exists with name '" + model_name + "'")
        return

    print('Beginning training...')
    # Define transformations for dataset
    transform = get_basic_transformation()

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)

    # Define loss function and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    # Perform training
    for epoch in range(25):

        running_loss = 0.0
        full_running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs,labels = data

            optimiser.zero_grad()

            
            outputs = model(inputs)
            
            if debug:
                print("\noutputs are:")
                print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimiser.step()

            # Print average loss every 2000 mini-batches
            running_loss += loss.item()
            full_running_loss += loss.item()
            if spiking:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, full_running_loss/(i+1)))
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    print("Finished training")

    model_path = "./models/" + model_name + ".pth"
    print("Saving to" + model_path)
    torch.save(model.state_dict(), model_path)



def test_CIFAR10(model, model_name, fun_trained = True):
    """
    Function to test a model that has been trained on CIFAR-10 image classification

    Args:
    model -- the class of model to test
    model_name -- the unique name of the trained model, used for loading from models/ folder
    fun_trained -- indicating if the model has been trained with the function above (and so is saved in that location)
    """
    print('Beginning testing...')
    torch.manual_seed(6926)
    # Define transformations for dataset
    transform = get_basic_transformation()

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 0)

    # Load trained model, if trained model is not supplied by the 'model' argument (as indicated by fun_trained)
    if fun_trained:
        model.load_state_dict(torch.load('./models/' + model_name + '.pth'))
    
    # Print results
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    return(100 * correct / total)