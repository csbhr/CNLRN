import os.path as osp
import torch
import torch.utils.data as data
import data.util as util


class ImageTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(ImageTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        img_paths_LQ = util.glob_file_list(self.LQ_root)
        img_paths_GT = util.glob_file_list(self.GT_root)
        assert len(img_paths_LQ) == len(img_paths_GT), 'Different number of images in LQ and GT folders'
        self.data_info['path_LQ'].extend(img_paths_LQ)
        self.data_info['path_GT'].extend(img_paths_GT)

        if self.cache_data:
            self.imgs_LQ = util.read_img_seq(img_paths_LQ)
            self.imgs_GT = util.read_img_seq(img_paths_GT)

    def __getitem__(self, index):

        if self.cache_data:
            imgs_LQ = self.imgs_LQ[index]
            img_GT = self.imgs_GT[index]
        else:
            pass  # TODO

        return {
            'LQ': imgs_LQ,
            'GT': img_GT,
            'LQ_path': self.data_info['path_LQ'][index],
            'GT_path': self.data_info['path_GT'][index]
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
