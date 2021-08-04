import torchvision

def get_basic_transformation(data = "cifar10"):
    """Return standard pytorch transformation for image input"""
    if data == "cifar10":
        t = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    elif data == "mnist":
        t = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

    return t