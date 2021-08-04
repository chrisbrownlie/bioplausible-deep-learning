from bindsnet.network import Network
import dill
import torch

with open("bindsnet.pickle", "rb") as dill_file:
    bindsnet_snn = dill.load(dill_file)
print("loaded ok")
bindsnet_snn.save("bindsnet_snn.pt")
print("torch.saved")