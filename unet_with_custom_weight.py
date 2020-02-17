#!/usr/bin/env python
# coding: utf-8

# In[9]:


import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import os
import random
import glob
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
from skimage import io
from skimage.exposure import histogram
from skimage.morphology import binary_dilation,binary_erosion,disk,square
from skimage.filters.rank import mean_bilateral
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
# Wandb init
import wandb
#wandb.init(project = "netra")

os.environ['WANDB_API_KEY'] = "my_api_key"

os.environ['WANDB_MODE'] = "dryrun"
# In[ ]:

wandb.init(project = "netra")
path_dir = '/scratch/netra/Datasets/Drive_Dataset/'


# In[ ]:


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[ ]:


import torch.utils.data as data

class DataLoaderSegmentation(data.Dataset):
    def __init__(self,folder_path,transform = None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.tif'))
        self.mask_files = glob.glob(os.path.join(folder_path,'new_mask','*.bmp'))
        self.alpha_files = glob.glob(os.path.join(folder_path,'alpha_mask','*gif'))
        self.transforms = transform
        #for img_path in img_files:
         #   self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path))
         
    def mask_to_class(self,mask):
        target = torch.from_numpy(mask)
        assert target.shape[2] ==3
        h,w = target.shape[0],target.shape[1]
        masks = torch.empty(h, w, dtype=torch.long)
        colors = torch.unique(target.view(-1,target.size(2)),dim=0).numpy()
        target = target.permute(2, 0, 1).contiguous()
        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
        for k in mapping:
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3) 
            masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
        return masks
    
    def elastic_transform_nearest(self,image, alpha=1000, sigma=20, spline_order=0, mode='nearest', random_state=np.random):
        
        image = np.array(image)
       # assert image.ndim == 3
        shape = image.shape[:2]

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                      sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        result = np.empty_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
        result = Image.fromarray(result)
        return result
    
    def elastic_transform_bilinear(self,image, alpha=1000, sigma=20, spline_order=1, mode='nearest', random_state=np.random):
        

        image = np.array(image)
        #assert image.ndim == 3
        shape = image.shape[:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        result = np.empty_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
        result = Image.fromarray(result)
        return result
    
    def gaussian_blur(self,img_dir,mask_dir):
        img = io.imread(img_dir,plugin = 'pil')
        mask = io.imread(mask_dir,plugin = 'pil')
        a = np.pad(img, ((100,100), (100,100), (0,0)), mode = "constant")
        img = a
        grayscale = rgb2gray(a)
        global_thresh = threshold_otsu(grayscale)
        binary_global1 = grayscale > global_thresh
        
        num_px_to_expand = 100
        # process each channel (RGB) separately
        for channel in range(a.shape[2]):

    # select a single channel
            one_channel = a[:, :, channel]

    # reset binary_global for the each channel
            binary_global = binary_global1.copy()

    # erode by 5 px to get rid of unusual edges from original image
            binary_global = binary_erosion(binary_global, disk(5))

    # turn everything less than the threshold to 0
            one_channel = one_channel * binary_global

    # update pixels one at a time
            for jj in range(num_px_to_expand):

        # get 1 px ring of to update
                px_to_update = np.logical_xor(binary_dilation(binary_global, disk(1)), 
                                      binary_global)

        # update those pixels with the average of their neighborhood
                x, y = np.where(px_to_update == 1)

                for x, y in zip(x,y):
            # make 3 x 3 px slices
                    slices = np.s_[(x-1):(x+2), (y-1):(y+2)]

            # update a single pixel
                    one_channel[x, y] = (np.sum(one_channel[slices]*
                                             binary_global[slices]) / 
                                       np.sum(binary_global[slices]))      


        # update original image
                a[:,:, channel] = one_channel

        # increase binary_global by 1 px dilation
                binary_global = binary_dilation(binary_global, disk(1))
            
            
            image_blur = cv2.GaussianBlur(a,(65,65),60)
            new_image = cv2.subtract(img,image_blur, dtype=cv2.CV_32F)
            out = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out = out[100:684,100:665,:]
            mask_bool = mask>0
            crop_img = a[100:684,100:665,:]
            gray = rgb2gray(crop_img)
            global_thresh = threshold_otsu(grayscale)
            binary_global_crop = gray > global_thresh
            px_to_update = np.logical_not(np.logical_and(binary_global_crop,mask_bool))
            x, y = np.where(px_to_update == 1)
            for x, y in zip(x,y):
                out[x,y,:] = 0
            
            out = Image.fromarray(out)
            return out

    def transform(self,image,mask):
        i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        #image = TF.Lambda(gaussian_blur),
       # mask = 
        #image = TF.Lambda(elastic_transform)
        # Random horizontal flipping
        #image = transforms.transforms.Lambda(gaussian_blur)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        image = TF.rotate(image,45)
        mask = TF.rotate(mask,45)
        image = TF.rotate(image,90)
        mask = TF.rotate(mask,90)
        image = TF.rotate(image,135)
        mask = TF.rotate(mask,135)
        image = TF.rotate(image,180)
        mask = TF.rotate(mask,180)
        image = TF.rotate(image,225)
        mask = TF.rotate(mask,225)     
        image = TF.rotate(image,270)
        mask = TF.rotate(mask,270)

        # Transform to tensor
        #image = TF.to_tensor(image)
#         mask = TF.to_tensor(mask)
        return image, mask
     
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        alpha_path = self.alpha_files[index]
        #data = Image.open(img_path)
        label = Image.open(mask_path)
       # label = np.array(label)
        data = self.gaussian_blur(img_path,alpha_path)
        data = self.elastic_transform_bilinear(data)
        label = self.elastic_transform_nearest(label)
        data,label = self.transform(data,label)
        label = np.array(label)
        data = np.array(data)
        #label = np.transpose(label,(2,0,1))
        mask = self.mask_to_class(label)
        if transforms is not None:
             data = self.transforms(data)
        return data,mask
       # return data, torch.from_numpy(label).long()
           
    def __len__(self):
        return len(self.img_files)


# In[ ]:


from skimage.segmentation import find_boundaries

w0 = 10
sigma = 5

def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.
    
    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """
    masks = masks.numpy()
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    ZZ = torch.from_numpy(ZZ)
    ZZ = ZZ.type(torch.float)
    ZZ = ZZ.cuda()
    return ZZ


# In[ ]:





# In[ ]:


def elastic_transform_bilinear(image, alpha=1000, sigma=20, spline_order=1, mode='constant', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
#     assert image.ndim == 3
    image = Image.open(image)
    image = np.array(image)
 #   assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    result = Image.fromarray(result)
    return result


# In[ ]:





# In[ ]:





# In[ ]:


def elastic_transform_nearest(image, alpha=1000, sigma=20, spline_order=0, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
#     assert image.ndim == 3
    image = np.array(image)
   # assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    result = Image.fromarray(result)
    return result


# In[ ]:





# In[ ]:


def elastic_transform_bilinear(image, alpha=1000, sigma=20, spline_order=1, mode='constant', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
#     assert image.ndim == 3
    
    image = np.array(image)
 #   assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    result = Image.fromarray(result)
    return result


# In[ ]:





# In[16]:



train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Gpu is being used")
else:
    print("No gpu")


# In[ ]:


batch_size = 5


# In[ ]:


dataset = DataLoaderSegmentation(path_dir,transform = transforms.ToTensor())
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)


# In[ ]:


for image,labels in iter(train_loader):
#     labels = labels.view(labels.shape[:2],-1)
    print(image.shape)
    print(labels.shape)
    break


# In[ ]:


train_mean = []
train_std = []

for i,image in enumerate(train_loader,0):
#      image[0].shape()
    numpy_image = image[0].numpy()
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std = np.std(numpy_image, axis=(0, 2, 3))
    
    train_mean.append(batch_mean)
    train_std.append(batch_std)
    
train_mean = torch.tensor(np.mean(train_mean, axis=0))
train_std = torch.tensor(np.mean(train_std, axis=0))

print('Mean:', train_mean)
print('Std Dev:', train_std)


# In[ ]:





# In[ ]:


import torchvision.transforms as transforms
data_transforms = transforms.Compose([
                                 #transforms.Lambda(gaussian_blur),
                                 #transforms.Lambda(elastic_transform),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=train_mean, std=train_std)
                               ])
final_dataset = DataLoaderSegmentation(path_dir,transform = data_transforms)


# In[ ]:


validation_split = 0.15
shuffle_dataset = True
random_seed = 42

dataset_size = len(final_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split*dataset_size))
if shuffle_dataset: 
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
train_indices, val_indices = indices[split:], indices[:split]


# In[ ]:


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


# In[ ]:


len(train_sampler),len(valid_sampler)


# In[ ]:


final_train_loader = DataLoader(dataset=final_dataset, batch_size = batch_size,sampler = train_sampler)
final_valid_loader = DataLoader(dataset=final_dataset, batch_size = batch_size,sampler = valid_sampler)


# In[ ]:


# Checking the dataset
for idx,(images, labels) in enumerate(final_train_loader):  
    print("Batch is: ",idx)
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# In[ ]:


for idx,(images, labels) in enumerate(final_valid_loader): 
    print("Batch is: ",idx)
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# In[ ]:





# In[2]:


import torch.nn as nn
import torch.nn.functional as F


# In[ ]:





# In[ ]:


# model = UNet()
# print(model)


# In[4]:


class Unet(nn.Module):
    def contracting_block(self,in_channels,out_channels,kernel_size = 5):
        block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels,eps = 1e-03, momentum = 0.99),
            nn.Conv2d(in_channels = out_channels,out_channels = out_channels,kernel_size = kernel_size, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels,eps = 1e-03, momentum = 0.99)
        )
        return block
      
    def expansive_block(self,in_channels,mid_channel,out_channels,kernel_size = 5):
        block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = mid_channel,kernel_size = kernel_size,padding = 2 ),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel,eps = 1e-03,momentum = 0.99),
            nn.Conv2d(in_channels = mid_channel, out_channels = mid_channel,kernel_size = kernel_size, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(in_channels = mid_channel,out_channels = out_channels,kernel_size = 2,stride = 2,padding = 0,output_padding=0)
        ) 
        return block
    
    def final_block(self,in_channels,mid_channels,out_channels,kernel_size = 5):
        block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = mid_channels,kernel_size = kernel_size,padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels,eps = 1e-03,momentum = 0.99),
            nn.Conv2d(in_channels = mid_channels,out_channels = mid_channels,kernel_size = kernel_size,padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels,eps = 1e-03,momentum = 0.99),
            nn.Conv2d(kernel_size = 1,in_channels=mid_channels, out_channels=out_channels),
           # nn.Softmax(),
            
        )    
        return block
    
    def __init__(self,in_channel,out_channel):
        super(Unet,self).__init__()
        self.conv_encode1 = self.contracting_block(in_channels = in_channel,out_channels = 16)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv_encode2 = self.contracting_block(16, 32)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv_encode3 = self.contracting_block(32, 64)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv_encode4 = self.contracting_block(64,128)
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.bottleneck = nn.Sequential(
                            nn.Conv2d(kernel_size=5, in_channels=128, out_channels=256,padding = 2),
                            nn.ReLU(),
                            nn.BatchNorm2d(256),
                            nn.Dropout(0.2),
                            nn.Conv2d(kernel_size=5, in_channels=256, out_channels=256, padding = 2),
                            nn.ReLU(),
                            nn.BatchNorm2d(256),
                            nn.Dropout(0.2),
                            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0)
                            )
        
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.conv_decode1 = self.expansive_block(64,32,16)
        self.final_layer = self.final_block(16, 16, out_channel)
        
               
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer3 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        cat_layer2 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode1(decode_block1)
        final_layer = self.final_layer(cat_layer1)
        return  final_layer
    


# In[5]:


unet = Unet(in_channel=3,out_channel=4)
print(unet)


# In[ ]:





# In[ ]:





# In[17]:
def weight_pre_calc(mask):
    label = Image.open(mask)
    target = np.array(label)
    target = torch.from_numpy(target)
    assert target.shape[2] ==3
    h,w = target.shape[0],target.shape[1]
    masks = torch.empty(h, w, dtype=torch.long)
    colors = torch.unique(target.view(-1,target.size(2)),dim=0).numpy()
    target = target.permute(2, 0, 1).contiguous()
    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
    for k in mapping:
        idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3) 
        masks[validx] = torch.tensor(mapping[k], dtype=torch.long)
    return masks

def cal_weights(folder_path):
    mask_files = glob.glob(os.path.join(folder_path,'new_mask','*.bmp'))
    weight_actual = torch.zeros([4], dtype=torch.float32)
    weight_normalized = torch.zeros([4], dtype=torch.float32)
    for msk in mask_files:
        mask = weight_pre_calc(msk)
        class0, class1 = mask.unique(return_counts = True)
        weight_actual = weight_actual + 1.0/class1.type(torch.float32)
        weights_normalized = weight_normalized + ((1.0 / class1) / (1.0 / class1).sum()).type(torch.float32)
        
    return weights_actual,weights_normalized
# Move models to CUDA
#device_id  = torch.cuda.device_count()

#if torch.cuda.device_count() == 1:
#    print("Let's use", torch.cuda.device_count(),"GPUs!")
#    unet = unet.to("cuda:{}".format(1))
#elif device_id ==2:
#    unet = nn.DataParallel(unet)
#    unet = unet.to("cuda:{}".format(2))
if train_on_gpu:
    unet = unet.cuda(0)
#if torch.cuda.device_count() > 1:
#    unet = nn.DataParallel(unet)
weights = torch.tensor([0.0015,0.0302,0.0371,0.9313])
weights = weights.cuda()
criterion = torch.nn.CrossEntropyLoss(weights)
optimizer = torch.optim.Adam(unet.parameters(), lr = 0.001)


# In[ ]:


def mask_to_class(mask):
    target = mask.cpu()
    h,w = target.shape[0],target.shape[1]
    masks = torch.empty(h,w,dtype = torch.long)
    colors = torch.unique(target.view(-1,target.size(2)),dim=0).numpy()
    target = target.permute(2,0,1).contiguous()
    mapping = {tuple(c): t for c,t in zip(colors.tolist(),range(len(colors)))}
    for k in mapping:
        idx = (target==torch.tensor(k,dtype= torch.long).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) ==3)
        masks[validx] = torch.tensor(mapping[k],dtype=torch.long)
    return masks


# In[ ]:


def precompute_for_images(masks):
    masks = masks.cpu()
    cls = masks.unique()
    res = torch.stack([torch.where(masks==cls_val,torch.tensor(1),torch.tensor(0)) for cls_val in cls])
    return res


# In[ ]:





# In[ ]:
import torch.nn.functional as F
import time
start = time.time()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(final_train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            temp_target = torch.squeeze(target)
            #temp_target = mask_to_class(temp_target)
            #temp_target = precompute_for_images(temp_target)
            #logp = F.log_softmax(output)
           # weights = make_weight_map(temp_target)
           # weighed_logp = (logp * weights).view(batch_size,-1)
           # weighed_loss = weighed_logp.sum(1)/weights.view(batch_size,-1).sum(1)
           # weighed_loss = -1 * weighed_loss.mean()
           # loss = weighed_loss
            #print("Input to weight map",temp_target.shape)
            #print("Target to model",target.shape)
            #print("Data to model",data.shape)
            #w = make_weight_map(temp_target)
            #loss = F.cross_entropy(output, target, w)
            loss = criterion(output,target)
           # loss = loss * w
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            ## Wandb log
       
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        test_loss = 0.0
        test_iou = 0.0
        for batch_idx, (data, target) in enumerate(final_valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output,target)
            valid_loss +=  valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
           

            ###  Testing ####
            target_counts = torch.zeros(4).cuda()
            pred_counts = torch.zeros(4).cuda()
            a, b = target.unique(return_counts = True)
            b = b.type(torch.float)
            target_counts[a] += b
            ## target_counts 
            _, predicted = torch.max(output.data,1)
            index, count = predicted.unique(return_counts = True)
            ## predicted counts
            #pred_counts[index] +=count
            count = count.type(torch.float)
            pred_counts[index] += count
            intersection0, intersection1, intersection2, intersection3 = (pred_counts.min(target_counts))
            intersection0 = intersection0.item()
            intersection1 = intersection1.item()
            intersection2 = intersection2.item()
            intersection3 = intersection3.item()
            union0 = ((pred_counts[0] + target_counts[0]) - intersection0).item()
            union1 = ((pred_counts[1] + target_counts[1]) - intersection1).item()
            union2 = ((pred_counts[2] + target_counts[2]) - intersection2).item()
            union3 = ((pred_counts[3] + target_counts[3]) - intersection3).item()
            iou0 = intersection0 / union0
            iou1 = intersection1 / union1
            iou2 = intersection2 / union2
            iou3 = intersection3 / union3
            avg_iou = (iou0 + iou1+iou2+ iou3) /4
            test_iou += ((1 / (batch_idx + 1)) * (avg_iou - test_iou))
        # print training/validation statistics 
        test_iou = test_iou * 100.
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValid Accuracy:{:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss,
            test_iou
            ))
        # Wandb Log
        wandb.log({'epoch':epoch,'training_loss':train_loss,'validation_loss':valid_loss,'test_accuracy': test_iou})
       # wandb.log('epoch':epoch,'training_loss':valid_loss)
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,valid_loss))
            torch.save(model.state_dict(), '/home/rxs1576/Experiments/triton_models/dict_model_2.pt')
            torch.save(model,'/home/rxs1576/Experiments/triton_models/com_model_2.pt')
            valid_loss_min = valid_loss
    # return trained model
    return model


# train the model
model_scratch = train(150, final_train_loader, unet, optimizer, 
                      criterion, train_on_gpu)
end = time.time()
print(end-start)

# In[64]:


# load the model that got the best validation accuracy
#model = Unet(in_channel=3,out_channel=4)
#model.load_state_dict(torch.load('/home/rxs1576/Experiments/model_scratch.pt', map_location=lambda storage, loc: storage))
#model.eval()


# In[ ]:





# In[ ]:





# In[65]:





# In[82]:





# In[83]:





# In[84]:





# In[104]:





# In[86]:





# In[87]:





# In[ ]:





# In[ ]:





# In[88]:





# In[105]:





# In[ ]:





# In[99]:



  


# In[100]:





# In[103]:





# In[89]:





# In[90]:





# In[91]:





# In[92]:





# In[93]:





# In[94]:





# In[95]:





# In[96]:





# In[97]:





# In[59]:





# In[60]:





# In[61]:





# In[62]:





# In[63]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




