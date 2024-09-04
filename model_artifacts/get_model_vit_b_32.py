import torch
from torchvision.models import vit_b_32

model = vit_b_32(weights='IMAGENET1K_V1')

model.eval()

model_path = "vit_b_32.pt"
torch.save(model.state_dict(), model_path)