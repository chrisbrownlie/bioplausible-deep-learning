import bindsnet.conversion.conversion as conversion
from baseline_model import *

converted_snn = conversion.ann_to_snn(simpleModel(), [3, 32, 32])

# Saving network
print("Saving network")
converted_snn.save("conversion_snn.pt")

print("Saving network state_dict")
torch.save(converted_snn.state_dict, "conversion_snn_sd.pt")