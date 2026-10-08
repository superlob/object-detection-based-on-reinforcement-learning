from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader
import os

root_dir = "dataset"
block_dir = "label"
image_size = 480
transform_img = transforms.Compose([transforms.Resize([image_size,image_size])])

class MyData(Dataset):    # 写子类继承Dataset
    # 初始化，定义全局变量
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.block_dir =  block_dir
        self.label_list = os.listdir(self.root_dir)
        self.label_list.sort()
        self.transform = transform_img
        self.image_name = 0

    def getitem(self, label, idx):
        #获取图片
        image_path = os.path.join(self.root_dir, self.label_list[label])
        image_list = os.listdir(image_path)
        self.image_name = image_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_list[label], self.image_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        label = label

        #获取框
        image_name_xml = self.image_name.split(".")[0]+".xml"
        block_xml_path = os.path.join(self.block_dir, image_name_xml)
        tree = ET.parse(block_xml_path)
        root = tree.getroot()
        #获取原图片和矩形框的尺寸
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        obj = root.find("object")
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        box = []
        #调整框的位置
        scale_x = image_size / width
        scale_y = image_size / height
        xmin_new = int(xmin * scale_x)
        ymin_new = int(ymin * scale_y)
        xmax_new = int(xmax * scale_x)
        ymax_new = int(ymax * scale_y)
        box = (xmin_new, ymin_new, xmax_new, ymax_new)

        return img,label,box

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

mydataset = MyData(root_dir,transform_img)
