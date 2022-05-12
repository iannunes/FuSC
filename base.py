import os
import sys
import numpy as np
import torch

from torch.utils import data

from skimage import io
from skimage import color
from skimage import measure
from skimage import transform
from skimage import util

from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors

import time

# Class that reads a sequence of image paths from a directory and creates a data.Dataset with them.
class ListDataset(data.Dataset):
    
    def __init__(self, dataset, mode, crop_size, normalization='minmax', hidden_classes=None, overlap=False, use_dsm=False, dataset_path='../'):
        
        # Initializing variables.
        self.root = dataset_path + dataset + '/'
        self.dataset = dataset
        self.mode = mode
        self.crop_size = crop_size
        self.normalization = normalization
        self.hidden_classes = hidden_classes
        self.overlap = overlap
        self.use_dsm = use_dsm
        
        if self.dataset == 'GRSS':
            self.num_classes = 21
        else:
            self.num_classes = 5
            
        if self.hidden_classes is not None:
            self.n_classes = self.num_classes - len(hidden_classes)
        else:
            self.n_classes = self.num_classes
            
        print('self.n_classes', self.n_classes)
        print('self.hidden_classes', self.hidden_classes)
        
        # Creating list of paths.
        if self.dataset == 'GRSS':
            
            self.make_dataset()
            
        else:
            
            self.imgs = self.make_dataset()
            
            # Check for consistency in list.
            if len(self.imgs) == 0:
                
                raise (RuntimeError('Found 0 images, please check the data set'))
                
    def make_dataset(self):
        
        # Making sure the mode is correct.
        assert self.mode in ['Train', 'Test', 'Validate']
        
        # Setting string for the mode.
        img_dir = os.path.join(self.root, self.mode, 'JPEGImages')
        msk_dir = os.path.join(self.root, self.mode, 'Masks')
        if self.use_dsm:
            dsm_dir = os.path.join(self.root, self.mode, 'NDSM')
                
        if self.dataset == 'GRSS':
            
            # Presetting ratios across GRSS channels and labels.
            self.rgb_hsi_ratio = 20
            self.dsm_hsi_ratio = 2
            self.msk_hsi_ratio = 2
            
            self.rgb_msk_ratio = 10
            
            self.hsi_patch_size = 500
            self.rgb_patch_size = self.rgb_hsi_ratio * self.hsi_patch_size
            self.dsm_patch_size = self.dsm_hsi_ratio * self.hsi_patch_size
            self.msk_patch_size = self.msk_hsi_ratio * self.hsi_patch_size
            
            if self.mode == 'Train' or self.mode == 'Validate':
                
                # Reading images.
                self.img_single = io.imread(os.path.join(self.root, 'Train', 'Images', 'rgb_clipped.tif')).astype(np.uint8)
                self.msk_single = io.imread(os.path.join(self.root, 'Train', 'Masks', '2018_IEEE_GRSS_DFC_GT_TR.tif')).astype(np.int64)
                if self.use_dsm:
                    self.dsm_single = io.imread(os.path.join(self.root, 'Train', 'DSM', 'dsm_clipped.tif'))
                    self.q0001 = -21.208378 # q0001 precomputed from training set.
                    self.q9999 = 41.01488   # q9999 precomputed from training set.
                    self.dsm_single = np.clip(self.dsm_single, self.q0001, self.q9999)
                    self.dsm_single = (self.dsm_single - self.dsm_single.min()) / (self.dsm_single.max() - self.dsm_single.min())
                    self.dsm_single *= 255.0
                    
            elif self.mode == 'Test':
                
                # Reading images.
                self.img_single = io.imread(os.path.join(self.root, 'Test', 'Images', 'rgb_merged.tif')).astype(np.uint8)
                self.msk_single = io.imread(os.path.join(self.root, 'Test', 'Masks', 'Test_Labels_osr.tif'))[:,:,0].astype(np.int64)
                self.msk_single[self.msk_single == 100] = 0
                if self.use_dsm:
                    self.dsm_single = io.imread(os.path.join(self.root, 'Test', 'DSM', 'UH17c_GEF051.tif'))
                    self.q0001 = -21.208378 # q0001 precomputed from training set.
                    self.q9999 = 41.01488   # q9999 precomputed from training set.
                    self.dsm_single = np.clip(self.dsm_single, self.q0001, self.q9999)
                    self.dsm_single = (self.dsm_single - self.dsm_single.min()) / (self.dsm_single.max() - self.dsm_single.min())
                    self.dsm_single *= 255.0
                
            self.msk_single, self.msk_true_single = self.shift_labels(self.msk_single)
            
            unique, counts = np.unique(self.msk_single, return_counts=True)
            print(unique)
            valid_counts = counts[:-1] if self.mode == 'Train' else counts[:-2] # Removing UUC.
            self.weights = (valid_counts.max() / valid_counts).tolist()
            print('weights', self.weights)
            
            print('img_single', self.img_single.shape)
            print('dsm_single', self.dsm_single.shape)
            print('msk_single', self.msk_single.shape)
            print('msk_true_single', self.msk_true_single.shape)
            
            return
        
        else:
            
            # Vaihingen and Potsdam.
            if self.mode == 'Validate':
                img_dir = os.path.join(self.root, 'ValidateTrain', 'JPEGImages')
                msk_dir = os.path.join(self.root, 'ValidateTrain', 'Masks')
                if self.use_dsm:
                    dsm_dir = os.path.join(self.root, 'ValidateTrain', 'NDSM')

            data_list = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

            # Creating list containing image and ground truth paths.
            items = []
            if self.dataset == 'Vaihingen':
                for it in data_list:
                    item = (
                        os.path.join(img_dir, it),
                        os.path.join(msk_dir, it),
                        os.path.join(dsm_dir, it.replace('top_mosaic_09cm_area', 'dsm_09cm_matching_area').replace('.tif', '_normalized.jpg'))
                    )
                    items.append(item)
            elif self.dataset == 'Potsdam':
                for it in data_list:
                    item = (
                        os.path.join(img_dir, it),
                        os.path.join(msk_dir, it.replace('_IRRG.tif', '_label_noBoundary.tif')),
                        os.path.join(dsm_dir, it.replace('top_potsdam_', 'dsm_potsdam_').replace('_IRRG.tif', '_normalized_lastools.jpg'))
                    )
                    items.append(item)
        
            # Returning list.
            return items
    
    def object_crops(self, img, msk, msk_true, n_crops):
        
        img_crop_list = []
        msk_crop_list = []
        msk_true_crop_list = []
        
        rand_fliplr = np.random.random() > 0.50
        rand_flipud = np.random.random() > 0.50
        rand_rotate = np.random.random()

        label_msk = measure.label(msk > 0)
        regions = measure.regionprops(label_msk)
        
        perm = np.random.permutation(len(regions))
        
        for i in range(n_crops):
            
            bbox = regions[perm[i % len(regions)]].bbox # (min_row, min_col, max_row, max_col)
            centroid = regions[perm[i % len(regions)]].centroid # (row, col)

            rand_y = np.random.randint(low=max(0, bbox[0] - (self.crop_size[0] // 2)), high=max(1, bbox[2] - (self.crop_size[0] // 2)))
            rand_x = np.random.randint(low=max(0, bbox[1] - (self.crop_size[1] // 2)), high=max(1, bbox[3] - (self.crop_size[1] // 2)))
            
            if rand_y + self.crop_size[0] > msk.shape[0]:
                rand_y = msk.shape[0] - self.crop_size[0]
            if rand_x + self.crop_size[1] > msk.shape[1]:
                rand_x = msk.shape[1] - self.crop_size[1]

            img_patch = img[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_patch = msk[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_true_patch = msk_true[rand_y:(rand_y + self.crop_size[0]),
                                      rand_x:(rand_x + self.crop_size[1])]
            
            if rand_fliplr:
                img_patch = np.fliplr(img_patch)
                msk_patch = np.fliplr(msk_patch)
                msk_true_patch = np.fliplr(msk_true_patch)
            if rand_flipud:
                img_patch = np.flipud(img_patch)
                msk_patch = np.flipud(msk_patch)
                msk_true_patch = np.flipud(msk_true_patch)
            
            if rand_rotate < 0.25:
                img_patch = transform.rotate(img_patch, 270, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 270, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 270, order=0, preserve_range=True)
            elif rand_rotate < 0.50:
                img_patch = transform.rotate(img_patch, 180, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 180, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 180, order=0, preserve_range=True)
            elif rand_rotate < 0.75:
                img_patch = transform.rotate(img_patch, 90, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 90, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 90, order=0, preserve_range=True)
                
            img_patch = img_patch.astype(np.float32)
            msk_patch = msk_patch.astype(np.int64)
            msk_true_patch = msk_true_patch.astype(np.int64)
            
            img_crop_list.append(img_patch)
            msk_crop_list.append(msk_patch)
            msk_true_crop_list.append(msk_true_patch)
        
        img = np.asarray(img_crop_list)
        msk = np.asarray(msk_crop_list)
        msk_true = np.asarray(msk_true_crop_list)
        
        return img, msk, msk_true
    
    def random_crops(self, img, msk, msk_true, n_crops):
        
        img_crop_list = []
        msk_crop_list = []
        msk_true_crop_list = []
        
        rand_fliplr = np.random.random() > 0.50
        rand_flipud = np.random.random() > 0.50
        rand_rotate = np.random.random()
        
        for i in range(n_crops):
            
            rand_y = np.random.randint(msk.shape[0] - self.crop_size[0])
            rand_x = np.random.randint(msk.shape[1] - self.crop_size[1])

            img_patch = img[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_patch = msk[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_true_patch = msk_true[rand_y:(rand_y + self.crop_size[0]),
                                      rand_x:(rand_x + self.crop_size[1])]
            
            if rand_fliplr:
                img_patch = np.fliplr(img_patch)
                msk_patch = np.fliplr(msk_patch)
                msk_true_patch = np.fliplr(msk_true_patch)
            if rand_flipud:
                img_patch = np.flipud(img_patch)
                msk_patch = np.flipud(msk_patch)
                msk_true_patch = np.flipud(msk_true_patch)
            
            if rand_rotate < 0.25:
                img_patch = transform.rotate(img_patch, 270, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 270, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 270, order=0, preserve_range=True)
            elif rand_rotate < 0.50:
                img_patch = transform.rotate(img_patch, 180, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 180, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 180, order=0, preserve_range=True)
            elif rand_rotate < 0.75:
                img_patch = transform.rotate(img_patch, 90, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 90, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 90, order=0, preserve_range=True)
                
            img_patch = img_patch.astype(np.float32)
            msk_patch = msk_patch.astype(np.int64)
            msk_true_patch = msk_true_patch.astype(np.int64)
            
            img_crop_list.append(img_patch)
            msk_crop_list.append(msk_patch)
            msk_true_crop_list.append(msk_true_patch)
        
        img = np.asarray(img_crop_list)
        msk = np.asarray(msk_crop_list)
        msk_true = np.asarray(msk_true_crop_list)
        
        return img, msk, msk_true
        
    def test_crops(self, img, msk, msk_true):
        
        n_channels = 3
        if self.use_dsm:
            n_channels = 4
        if self.overlap:
            w_img = util.view_as_windows(img,
                                         (self.crop_size[0], self.crop_size[1], n_channels),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2, n_channels)).squeeze()
            w_msk = util.view_as_windows(msk,
                                         (self.crop_size[0], self.crop_size[1]),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2))
            w_msk_true = util.view_as_windows(msk_true,
                                              (self.crop_size[0], self.crop_size[1]),
                                              (self.crop_size[0] // 2, self.crop_size[1] // 2))
        else:
            w_img = util.view_as_blocks(img, (self.crop_size[0], self.crop_size[1], n_channels)).squeeze()
            w_msk = util.view_as_blocks(msk, (self.crop_size[0], self.crop_size[1]))
            w_msk_true = util.view_as_blocks(msk_true, (self.crop_size[0], self.crop_size[1]))
        
        return w_img, w_msk, w_msk_true
        
    def shift_labels(self, msk):
        
        msk_true = np.copy(msk)
            
        if self.dataset == 'Vaihingen' or self.dataset == 'Potsdam':
            
            # Shifting clutter/background to unknown on labels.
            msk[msk == 5] = 2000
            msk[msk == 6] = 2000
        
        cont = 0
        for h_c in sorted(self.hidden_classes):
            
#             print('Hidden %d' % (h_c))
            msk[msk == h_c - cont] = 1000
            for c in range(h_c - cont + 1, self.num_classes):
#                 print('    Class %d -> %d' % (c, c - 1))
                msk[msk == c] = c - 1
                msk_true[msk_true == c] = c - 1
            cont = cont + 1
            
        msk_true[msk == 1000] = self.num_classes - len(self.hidden_classes)
        msk_true[msk == 2000] = self.num_classes
        msk[msk >= 1000] = self.num_classes - len(self.hidden_classes)
        
        #print('msk after', np.unique(msk))
        #print('msk_true after', np.unique(msk_true))
        
        return msk, msk_true
    
    def mask_to_class(self, msk):
        
        msk = msk.astype(np.int64)
        new = np.zeros((msk.shape[0], msk.shape[1]), dtype=np.int64)
        
        msk = msk // 255
        msk = msk * (1, 7, 49)
        msk = msk.sum(axis=2)
        
        new[msk == 1 + 7 + 49] = 0 # Street.
        new[msk ==         49] = 1 # Building.
        new[msk ==     7 + 49] = 2 # Grass.
        new[msk ==     7     ] = 3 # Tree.
        new[msk == 1 + 7     ] = 4 # Car.
        new[msk == 1         ] = 5 # Surfaces.
        new[msk == 0         ] = 6 # Boundaries.
        
        return new
        
    def __getitem__(self, index):
        
        img_raw = None
        msk_raw = None
        dsm_raw = None
        
        if self.dataset == 'GRSS':
            
            if self.mode == 'Train':

                offset_rgb = np.random.randint(self.rgb_msk_ratio, size=2) if self.mode == 'Train' else (self.rgb_msk_ratio // 2, self.rgb_msk_ratio // 2)
                
                img_raw = self.img_single[offset_rgb[0]::self.rgb_msk_ratio,
                                          offset_rgb[1]::self.rgb_msk_ratio]
                msk_raw = self.msk_single
                msk_true_raw = self.msk_true_single
                if self.use_dsm:
                    dsm_raw = self.dsm_single
                
                assert img_raw.shape[0] == dsm_raw.shape[0] and\
                       img_raw.shape[0] == msk_raw.shape[0] and\
                       img_raw.shape[0] == msk_true_raw.shape[0] and\
                       img_raw.shape[1] == dsm_raw.shape[1] and\
                       img_raw.shape[1] == msk_raw.shape[1] and\
                       img_raw.shape[1] == msk_true_raw.shape[1], 'Shape Inconsistency: rgb = ' + str(img_raw.shape) + ', dsm = ' + str(dsm_raw.shape) + ', msk = ' + str(msk_raw.shape) + ', msk_true = ' + str(msk_true_raw.shape)
                
            else:
                
                img_raw = self.img_single[self.rgb_msk_ratio // 2::self.rgb_msk_ratio,
                                          self.rgb_msk_ratio // 2::self.rgb_msk_ratio]
                msk_raw = self.msk_single
                msk_true_raw = self.msk_true_single
                if self.use_dsm:
                    dsm_raw = self.dsm_single
                    
                assert img_raw.shape[0] == dsm_raw.shape[0] and\
                       img_raw.shape[0] == msk_raw.shape[0] and\
                       img_raw.shape[0] == msk_true_raw.shape[0] and\
                       img_raw.shape[1] == dsm_raw.shape[1] and\
                       img_raw.shape[1] == msk_raw.shape[1] and\
                       img_raw.shape[1] == msk_true_raw.shape[1], 'Shape Inconsistency: rgb = ' + str(img_raw.shape) + ', dsm = ' + str(dsm_raw.shape) + ', msk = ' + str(msk_raw.shape) + ', msk_true = ' + str(msk_true_raw.shape)
            
        else:
            
            # Reading items from list.
            if self.use_dsm:
                img_path, msk_path, dsm_path = self.imgs[index]
            else:
                img_path, msk_path = self.imgs[index]
            
            # Reading images.
            img_raw = io.imread(img_path)
            msk_raw = io.imread(msk_path)
            if self.use_dsm:
                dsm_raw = io.imread(dsm_path)
            
        if len(img_raw.shape) == 2:
            img_raw = color.gray2rgb(img_raw)
        
        if self.use_dsm:
            img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                           img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                           img_raw.shape[2] + 1),
                          fill_value=0.0,
                          dtype=np.float32)
        else:
            img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                           img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                           img_raw.shape[2]),
                          fill_value=0.0,
                          dtype=np.float32)

        img[:img_raw.shape[0], :img_raw.shape[1], :img_raw.shape[2]] = img_raw
        if self.use_dsm:
            img[:dsm_raw.shape[0], :dsm_raw.shape[1], -1] = dsm_raw
        
        if self.dataset == 'GRSS':
            
            msk = np.full((msk_raw.shape[0] + self.crop_size[0] - (msk_raw.shape[0] % self.crop_size[0]),
                           msk_raw.shape[1] + self.crop_size[1] - (msk_raw.shape[1] % self.crop_size[1])),
                          fill_value=0,
                          dtype=np.int64)
            msk[:msk_raw.shape[0], :msk_raw.shape[1]] = msk_raw
            
            msk_true = np.full((msk_true_raw.shape[0] + self.crop_size[0] - (msk_true_raw.shape[0] % self.crop_size[0]),
                                msk_true_raw.shape[1] + self.crop_size[1] - (msk_true_raw.shape[1] % self.crop_size[1])),
                               fill_value=0,
                               dtype=np.int64)
            msk_true[:msk_true_raw.shape[0], :msk_true_raw.shape[1]] = msk_true_raw
            
        else:
            
            msk = np.full((msk_raw.shape[0] + self.crop_size[0] - (msk_raw.shape[0] % self.crop_size[0]),
                           msk_raw.shape[1] + self.crop_size[1] - (msk_raw.shape[1] % self.crop_size[1]),
                           msk_raw.shape[2]),
                          fill_value=0,
                          dtype=np.int64)
            msk[:msk_raw.shape[0], :msk_raw.shape[1]] = msk_raw
            
            msk = self.mask_to_class(msk)
            
            msk, msk_true = self.shift_labels(msk)
        
        # Normalization.
        img = (img / 255) - 0.5
        
        if self.mode == 'Train':
            
            if self.dataset == 'GRSS':
                img, msk, msk_true = self.random_crops(img, msk, msk_true, 4)
            else:
                img, msk, msk_true = self.random_crops(img, msk, msk_true, 4)
            
            img = np.transpose(img, (0, 3, 1, 2))
        
        elif self.mode == 'Validate':
            
            img, msk, msk_true = self.test_crops(img, msk, msk_true)
            
            img = np.transpose(img, (0, 1, 4, 2, 3))
            msk = np.transpose(msk, (0, 1, 2, 3))
            msk_true = np.transpose(msk_true, (0, 1, 2, 3))
        
        elif self.mode == 'Test':
            
            img, msk, msk_true = self.test_crops(img, msk, msk_true)
            
            img = np.transpose(img, (0, 1, 4, 2, 3))
            msk = np.transpose(msk, (0, 1, 2, 3))
            msk_true = np.transpose(msk_true, (0, 1, 2, 3))
        
#         if self.dataset == 'Vaihingen' or self.dataset == 'Potsdam':
            
#             msk[msk == self.num_classes + 1] = self.num_classes
#             msk_true[msk_true == self.num_classes + 1] = self.num_classes
        
        if self.dataset != 'GRSS':
            # Splitting path.
            spl = img_path.split('/')
        
        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)
        msk_true = torch.from_numpy(msk_true)

        if self.dataset == 'GRSS':
        
            # Returning to iterator.
            return img, msk, msk_true

        else:
            
            # Returning to iterator.
            return img, msk, msk_true, spl[-1]

    def __len__(self):
        if self.dataset == 'GRSS':
            return 1
        else:
            return len(self.imgs)

def save_cm(cm_list, cm_path, total_segments):
    cm_file = open(cm_path, 'w')
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, cm in enumerate(cm_list):
        cm_file.write(str(thresholds[i])+':\n')
        cm_file.write(str(cm).replace(' [', '').replace('[','').replace(']',''))
        cm_file.write('\n')
    cm_file.write('Total segments: %s' % (str(total_segments)))
    cm_file.close()


def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 np.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]


def trim_coords(img):

    # Mask of non-black pixels (assuming image has a single channel).
    bin = img > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(bin)

    # Bounding box of non-black pixels.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    return y0, y1, x0, x1


def get_thresholds(scr_open, msk, n_known):
    
    # Trimming and selecting valid samples.
    msk = msk.ravel()
    
    scr_open = scr_open.ravel()
    scr_open = scr_open[msk < (n_known + 1)]
    
    msk = msk[msk < (n_known + 1)]
    
    bin_msk = (msk == n_known)
    
    # Computing ROC and AUC.
    #print('    Computing Open thresholds...')
    fpr_open, tpr_open, t_open = metrics.roc_curve(bin_msk, scr_open)

    thresholds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    ind_open = []
    for t in thresholds:
        if t == 1.00:
            ind_open.append(-1)
        else:
            for i in range(len(t_open)):
                if tpr_open[i] >= t:
                    ind_open.append(i)
                    break
                    
    #print('    Computing Open AUC...')
    auc_open = metrics.roc_auc_score(bin_msk, scr_open)
    
    roc_open = (t_open, tpr_open, fpr_open, ind_open, auc_open)
    
    return roc_open


def get_curr_metric(msk, prd, n_known):
    
    tru_np = msk.ravel()
    prd_np = prd.ravel()
    
    tru_valid = tru_np[tru_np < (n_known + 1)]
    prd_valid = prd_np[tru_np < (n_known + 1)]

    #print('        Computing CM...')
    cm = metrics.confusion_matrix(tru_valid, prd_valid)

    #print('        Computing Accs...')
    tru_known = 0.0
    sum_known = 0.0

    for c in range(n_known):
        tru_known += float(cm[c, c])
        sum_known += float(cm[c, :].sum())

    acc_known = float(tru_known) / float(sum_known)
    
    tru_unknown = float(cm[n_known, n_known])
    sum_unknown_real = float(cm[n_known, :].sum())
    sum_unknown_pred = float(cm[:, n_known].sum())
    
    pre_unknown = 0.0
    rec_unknown = 0.0
    
    if sum_unknown_pred != 0.0:
        pre_unknown = float(tru_unknown) / float(sum_unknown_pred)
    if sum_unknown_real != 0.0:
        rec_unknown = float(tru_unknown) / float(sum_unknown_real)
        
    acc_unknown = (tru_known + tru_unknown) / (sum_known + sum_unknown_real)
    
    acc_mean = (acc_known + acc_unknown) / 2.0
    
    #print('        Computing Balanced Acc...')
    bal = metrics.balanced_accuracy_score(tru_valid, prd_valid)
    
    #print('        Computing Kappa...')
    kap = metrics.cohen_kappa_score(tru_valid, prd_valid)
    
    curr_metrics = [acc_known, acc_unknown, pre_unknown, rec_unknown, bal, kap]
    
    return curr_metrics, cm


def get_metrics(msk, prd_open, n_known):
    
    open_metrics, cm = get_curr_metric(msk, prd_open, n_known)
    
    return open_metrics, cm


def generate_roc_all(src, msk, roc_output_path, n_known):
    full_scr_open = np.concatenate(src)
    full_msk = np.concatenate(msk)

    roc_open = get_thresholds(full_scr_open, full_msk, n_known)
    #print(roc_open[0][7])

    # Plotting ROC.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)

    lw = 2

    ax.plot(roc_open[2], roc_open[1], color='deepskyblue', lw=lw, label='AUC Open: %0.3f' % roc_open[-1])

    ax.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.set_xlabel('FPR', size=25)
    ax.set_ylabel('TPR', size=25)

    ax.legend(loc='lower right', prop={'size': 25})

    plt.tight_layout()
    fig.savefig(roc_output_path)