import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
import os
from torchvision import transforms


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, imgset="train", scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        self.classnames=['negative', 'positive']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.imgset = imgset
        self.set_ = imgset
        self.filepaths = []
        # self.root_dir+'/'+set_+'/'+self.classnames[i]
        for i in range(len(self.classnames)):
            if self.set_ == "test":
                imgdir = self.root_dir+'/'+self.set_+'/**/'
            else:
                imgdir = os.path.join(self.root_dir,self.set_,self.classnames[i])
            imgdir = os.path.join(imgdir, "*.png")
            all_files = sorted(glob.glob(imgdir))
            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
            if self.set_ == "test":
                break
        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        if self.set_ is not "test":
            class_name = path.split('/')[-3]
            class_id = self.classnames.index(class_name)
        else:
            class_id = ""
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        
        return torch.stack(imgs), class_id, self.filepaths[idx*self.num_views:(idx+1)*self.num_views]
        
        # return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, imgset="train", scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12):
        self.classnames=['chair']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode  
        self.imgset = imgset

        set_ = imgset
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(root_dir+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)

