import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pathlib
#CNN Network

class ConvNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvNet,self).__init__()

        #Output size after convolution filter
        #((w-f+2P)/s) +1

        #Input shape= (256,3,256, 256)

        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,256, 256)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,256, 256)
        self.relu1=nn.ReLU()
        #Shape= (256,12,256, 256)

        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,128,128)


        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,128,128)
        self.relu2=nn.ReLU()
        #Shape= (256,20,128,128)



        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,128,128)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,128,128)
        self.relu3=nn.ReLU()
        #Shape= (256,32,128,128)


        self.fc=nn.Linear(in_features=128 * 128 * 32,out_features=num_classes)



        #Feed forwad function

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)

        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)


            #Above output will be in matrix form, with shape (256,32,128, 128)

        output=output.view(-1,32*128*128)


        output=self.fc(output)

        return output

# Oldingi model arxitekturasini qayta aniqlash
model = ConvNet(num_classes=10)

# GPU mavjud bo'lsa, yuklang
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Saqlangan og'irliklarni yuklash
model.load_state_dict(torch.load(r'C:\Users\mirja\myenv\DjangoAPI\selxoz_project\colab_code\best_checkpoint.model'))
model.eval()  # Modelni baholash rejimiga o'tkazish
print("Model yuklandi va baholash rejimida!")

from PIL import Image

# Tasvirni yuklash va transformatsiya qilish
image_path = r"C:\Users\mirja\myenv\DjangoAPI\selxoz_project\colab_code\test_images\Bakterienfruchtflecken_Tomate_Blatt_Xanthomonas_vesicatoria.jpg"
image = Image.open(image_path)

# Transformatsiyani modelga moslashtirish
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Tasvirni transformatsiya qilish
image_tensor = transformer(image).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

# Bashorat qilish
image_tensor = image_tensor.to(device)
output = model(image_tensor)

# Eng yuqori ehtimollikni topish
_, prediction = torch.max(output.data, 1)

# Sinflarni yuklash
dataset_path_local = r'C:\Users\mirja\myenv\DjangoAPI\selxoz_project\edited_image'
classes = sorted([j.name.split('/')[-1] for j in pathlib.Path(dataset_path_local).iterdir()])

# Bashorat qilingan sinfni ko'rsatish
print(f"Bashorat: {classes[prediction]}")
