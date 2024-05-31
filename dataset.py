from torch.utils.data import Dataset
import glob
from torchvision import transforms
from PIL import Image


tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Mydata(Dataset):
    def __init__(self,img,label):
        self.img = glob.glob(f'png/{img}/*')
        self.label = glob.glob(f'png/{label}/*')
    def __len__(self):
        return len(self.img)
    def __getitem__(self, idx):
        img_path = self.img[idx]
        label_path = self.label[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        return tf(img),tf(label)

if __name__ == '__main__':
    train_data = Mydata('train','train_labels')
    for i in train_data:
        print(i[0].shape,i[1].shape)


