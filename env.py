import torch
import torch.nn as nn
from data import mydataset, ReplayBuffer
from PIL import Image,ImageDraw,ImageFont
import torchvision
import os
from torchvision import transforms


test_dir = 'test_exp'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

#每次的平移距离，缩放比例
step_stride = 24
step_rate = 0.15

state_temp = torch.empty(25088)

iou_step_th = 0.005  #如果iou的波动在0.002以内，认为无明显变化，不给予奖励
iou_done_th = 0.64
reward_iou_step = 1  #每一步的iou奖励
reward_dist_step = 0.5  #每一步的中点距离奖励
reward_ratio_step = 0.5
reward_done = 8

image_height = 480 - 1

#对框住的图像进行裁剪，转换totensor
def tranform_block(img,box):
    image_tensor= img
    cropped = image_tensor.crop(box)
    transform_img = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
    otp = transform_img(cropped)
    return otp

#对hook进行定义，利用hook机制对图像中框住的部分进行特征提取,把特征图取出来
def forward_hook(module, inp, outp):
    fmap_block = outp
    otp = torch.flatten(outp)
    state_temp = otp
    print(state_temp.shape)

'''
    fmap_block = fmap_block[0, :16, :,:]
    print(fmap_block.shape)
    plt.figure(figsize=(10, 10))
    for index in range(16):
        plt.subplot(4, 4, index + 1, )
        plt.imshow(fmap_block[index].detach().numpy())
    plt.show()
'''

#init函数还没写完
class ObjectDetectionEnv:
    def __init__(self):
        self.dataset =mydataset
        self.image,self.label,self.target_box = self.dataset.getitem(0,0)
        self.current_box = [100, 100, 200, 200]
        self.last_box = [0,0,0,0]
        self.last_iou = self.compute_iou()
        self.last_dist = self.distance()
        self.last_ratio = self.ratio()

    def get_new_pic(self,count):
        self.image, self.label, self.target_box = self.dataset.getitem(0, count)
        self.last_iou = self.compute_iou()
        self.last_dist = self.distance()

    #状态空间建模
    def get_state(self):
        state = tranform_block(self.image,self.current_box)
        '''
        vgg16(input)
        state = state_temp
        '''
        return state

    #动作空间建模
    def step(self,action,done):
        self.last_box = self.current_box.copy()
        if action == 0:  # 左移
            if self.current_box[0]-step_stride >= 0:#如果没有超过边界
                self.current_box[0] -= step_stride
                self.current_box[2] -= step_stride
            else:
                self.current_box[0] = 0
                self.current_box[2] -= self.current_box[0]

        elif action == 1:  # 右移
            if self.current_box[2] + step_stride <= image_height:
                self.current_box[0] += step_stride
                self.current_box[2] += step_stride
            else:
                self.current_box[2] = image_height
                self.current_box[0] += image_height-self.current_box[2]

        elif action == 2:  # 上移
            if self.current_box[1] - step_stride >= 0:
                self.current_box[1] -= step_stride
                self.current_box[3] -= step_stride
            else:
                self.current_box[1] = 0
                self.current_box[3] -= self.current_box[1]

        elif action == 3:  # 下移
            if self.current_box[3] +step_stride <=image_height:
                self.current_box[1] += step_stride
                self.current_box[3] += step_stride
            else:
                self.current_box[3] = image_height
                self.current_box[1] += image_height-self.current_box[3]

        elif action == 4: # 等比例压缩
            dx = (self.current_box[2]-self.current_box[0])*0.1/2.0
            dy = (self.current_box[2]-self.current_box[0])*0.1/2.0
            self.current_box[0] += dx
            self.current_box[2] -= dx
            self.current_box[1] += dy
            self.current_box[3] -= dy

        elif action == 5: # 等比例扩张
            dx = (self.current_box[2]-self.current_box[0])*step_rate/2.0
            dy = (self.current_box[2]-self.current_box[0])*step_rate/2.0
            new_left = self.current_box[0] - dx
            new_right = self.current_box[2] + dx
            new_top = self.current_box[1] - dy
            new_bottom = self.current_box[3] + dy
            if new_left >= 0 and new_right <= image_height and new_top >= 0 and new_bottom <= image_height:
                self.current_box[0] = new_left
                self.current_box[2] = new_right
                self.current_box[1] = new_top
                self.current_box[3] = new_bottom
            else:
                self.current_box[0] = max(0, new_left)
                self.current_box[2] = min(image_height, new_right)
                self.current_box[1] = max(0, new_top)
                self.current_box[3] = min(image_height, new_bottom)

        elif action == 6:  # 左右扩张
            dx = (self.current_box[2] - self.current_box[0]) * step_rate / 2.0
            new_left = self.current_box[0] - dx
            new_right = self.current_box[2] + dx
            if new_left >= 0 and new_right <= image_height:
                self.current_box[0] = new_left
                self.current_box[2] = new_right
            else:
                # 如果超出边界，调整到最大可能范围
                self.current_box[0] = max(0, new_left)
                self.current_box[2] = min(image_height, new_right)

        elif action == 7:  # 左右压缩
            dx = (self.current_box[2] - self.current_box[0]) * step_rate / 2.0
            self.current_box[0] += dx
            self.current_box[2] -= dx

        elif action == 8:  # 上下扩张
            dy = (self.current_box[2] - self.current_box[0]) * step_rate / 2.0
            new_top = self.current_box[1] - dy
            new_bottom = self.current_box[3] + dy
            if new_top >= 0 and new_bottom <= image_height:
                self.current_box[1] = new_top
                self.current_box[3] = new_bottom
            else:
                self.current_box[1] = max(0, new_top)
                self.current_box[3] = min(image_height, new_bottom)

        elif action == 9:  # 上下压缩
            dy = (self.current_box[2] - self.current_box[0]) * step_rate / 2.0
            self.current_box[1] += dy
            self.current_box[3] -= dy

        reward,done = self.reward_function(done)
        state = self.get_state()
        return state,reward,done


    #奖励函数设计：考虑iou,曼哈顿距离，和高宽比
    def reward_function(self,done):
        reward = 0
        #iou
        iou = self.compute_iou()
        if done == False:  #如果还没到最后一步
            if iou < iou_done_th:
                d_iou = iou - self.last_iou
                print(d_iou)
                if d_iou > iou_step_th:
                    reward1 = reward_iou_step
                elif d_iou < -iou_step_th:
                    reward1 = -reward_iou_step
                else :
                    reward1 = 0
                reward += reward1
            else:
                reward += reward_done
                done = True
        else:
            if iou > iou_done_th:
                reward += reward_done
            elif iou < iou_done_th-0.1: #如果iou<0.5，认为检测不成功
                reward += -reward_done
            else:
                reward += 0.5*reward_done
        self.last_iou = iou
        #曼哈顿距离
        dist = self.distance()
        if done == False and reward1 == 0:
            if dist - self.last_dist <= -0.5*step_stride:
                reward2 = reward_dist_step
            elif dist - self.last_dist >= 0.5*step_stride:
                reward2 = -reward_dist_step
            else:
                reward2 = 0
            reward += reward2
        self.last_dist = dist

        #高宽比
        ratio = self.ratio()
        print(ratio - self.last_ratio)
        if done == False and reward1 == 0:
            if  ratio < self.last_ratio:
                reward3 = reward_ratio_step
            elif ratio > self.last_ratio:
                reward3 = -reward_ratio_step
            else:
                reward3 = 0
            reward += reward3
        self.last_ratio = ratio
        return reward,done

    def compute_iou(self):
        box1 = self.current_box
        box2 = self.target_box
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    #曼哈顿距离 越小离目标越接近
    def distance(self):
        x1 = (self.current_box[0]+self.current_box[2])/2
        y1 = (self.current_box[1]+self.current_box[3])/2
        x2 = (self.target_box[0]+self.target_box[2])/2
        y2 = (self.target_box[1]+self.target_box[3])/2
        return abs(x1 - x2) + abs(y1 - y2)

    #高宽比 越小离目标越接近
    def ratio(self):
        rt = (self.current_box[3]-self.current_box[1])/(self.current_box[2]-self.current_box[0])
        rg = (self.target_box[3]-self.target_box[1])/(self.target_box[2]-self.target_box[0])
        return abs(rt-rg)

    def draw_block(self,reward,i):
        image_PIL = self.image.copy()
        draw = ImageDraw.Draw(image_PIL)
        #draw.text((160, 20), f"reward:{reward}",  fill=(0, 0, 0,255))
        draw.rectangle(self.last_box,outline="white", width=2)
        draw.rectangle(self.current_box, outline="blue", width=2)
        draw.rectangle(self.target_box,outline="red",width=2)
        image_PIL.save(os.path.join(test_dir,f"{i}.jpg"))

    def reset(self):
        self.current_box = [0.25*image_height, 0.25*image_height, 0.75*image_height,0.75*image_height]
        self.last_iou = self.compute_iou()
        self.last_dist = self.distance()
        #self.last_ratio = self.ratio()
        return self.get_state()


#加载vgg16模型
'''
vgg16 = torchvision.models.vgg16(pretrained = True)
vgg16.features[30].register_forward_hook(forward_hook)
'''

env = ObjectDetectionEnv()






