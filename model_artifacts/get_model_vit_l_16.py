import torch
from torchvision.models import vit_l_16

model = vit_l_16(weights='IMAGENET1K_V1')

model.eval()

model_path = "vit_l_16.pt"
torch.save(model.state_dict(), model_path)