"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""

import sys
import os.path as osp
import pickle
from multiprocessing import Pool
import lmdb
import cv2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util as util  # noqa: E402


def main():
    mode = 'blur_bicubic_X4'
    img_folder = '../dataset/REDS/train_240/blur_bicubic_X4'
    lmdb_save_path = '../dataset/REDS/lmdb/train_240_blur_bicubic_X4.lmdb'
    H_dst, W_dst = 180, 320

    # mode = 'sharp_bicubic_X4'
    # img_folder = '../dataset/REDS/train_240/sharp_bicubic_X4'
    # lmdb_save_path = '../dataset/REDS/lmdb/train_240_sharp_bicubic_X4.lmdb'
    # H_dst, W_dst = 180, 320

    # mode = 'sharp'
    # img_folder = '../dataset/REDS/train_240/sharp'
    # lmdb_save_path = '../dataset/REDS/lmdb/train_240_sharp.lmdb'
    # H_dst, W_dst = 720, 1280

    REDS(mode, img_folder, lmdb_save_path, H_dst, W_dst)


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def REDS(mode, img_folder, lmdb_save_path, H_dst, W_dst):
    """Create lmdb for the REDS dataset, each image with a fixed size
    GT: [3, 720, 1280], key: 000_00000000
    LR: [3, 180, 320], key: 000_00000000
    key: 000_00000000

    flow: downsampled flow: [3, 360, 320], keys: 000_00000005_[p2, p1, n1, n2]
        Each flow is calculated with the GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(img_folder)
    keys = []
    for img_path in all_img_list:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        keys.append(folder + '_' + img_name)

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'REDS_{}_wval'.format(mode)
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    main()
