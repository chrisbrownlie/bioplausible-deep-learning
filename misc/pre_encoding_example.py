import torch
import torchvision
from norse.torch.functional.encode import poisson_encode
import matplotlib.pyplot as plt

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root = './data', transform = transform, train = True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 0)

for i,data in enumerate(trainloader, 0):
    if i == 3:
        break

    image,label = data
    print(label)
    plt.imshow(image[0].permute(1, 2, 0))
    #plt.show()
    plt.savefig('./plots/cifar_image' + str(i) + '.png')
    plt.clf()

    example_spikes = poisson_encode(image, 100, f_max = 20).reshape(100,3*32*32).to_sparse().coalesce()

    t = example_spikes.indices()[0]
    n = example_spikes.indices()[1]

    plt.scatter(t, n, marker='|', color='black')
    plt.ylabel('Input Unit')
    plt.xlabel('Time [ms]')
    #plt.show()
    
    plt.savefig('./plots/cifar_image' + str(i) + '_spikes.png')
    plt.clf()