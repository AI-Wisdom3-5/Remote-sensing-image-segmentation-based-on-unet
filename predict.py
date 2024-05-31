from model import *
from PIL import Image
from dataset import *
import numpy as np
import cv2
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path =
img = Image.open(path)
img = tf(img)
img = img.unsqueeze(0)
img = img.to(device)
model = Unet().to(device)
model.load_state_dict(torch.load('save_model/best_model.pt'))
model.eval()

out = model(img)

out = torch.squeeze(out)
out = np.transpose(out.cpu().detach().numpy(), (1, 2, 0))
# cv2.imshow('out', out)
# cv2.waitKey(0)
image   = Image.fromarray(np.uint8(out))
img = Image.open(path)
img = img.resize((128, 128))
img = Image.blend(img, image, 0.5)
save_image(img,'test_img/out.jpg')