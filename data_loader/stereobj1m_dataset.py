import torch.utils.data as data
import numpy as np
import os
import json
import cv2
from PIL import Image
import data_utils
from augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch


def resize_binary_map(binary_map, size):
    binary_map_tmp = []
    binary_map_shape = binary_map.shape
    if len(binary_map_shape) == 2:
        binary_map = np.expand_dims(binary_map, -1)
    for i in range(binary_map.shape[-1]):
        bm = binary_map[:, :, i]
        bm = cv2.resize(bm.astype('uint8'), size, \
                interpolation=cv2.INTER_NEAREST).astype('bool')
        binary_map_tmp.append(bm)
    binary_map = np.stack(binary_map_tmp, axis=-1)
    if len(binary_map_shape) == 2:
        binary_map = np.squeeze(binary_map)
    return binary_map


class Dataset(data.Dataset):

    def __init__(self, args, lr=False, transforms=None):
        super(Dataset, self).__init__()

        self.args = args
        self.height = args.image_height
        self.width = args.image_width
        self.lr = lr
        self.split = args.split

        self.load_cam_params()

        self.rotate_min = -30
        self.rotate_max = 30
        self.resize_ratio_min = 0.8
        self.resize_ratio_max = 1.2
        self.overlap_ratio = 1.0

        self.stereobj_root = args.data
        self.stereobj_data_root = self.stereobj_root
        self.cls_type = args.cls_type

        kp_filename = os.path.join(self.stereobj_root, 'objects', self.cls_type + '.kp')
        with open (kp_filename, 'r') as f:
            self.kps = f.read().split()
            self.kps = np.array([float(k) for k in self.kps])
            self.kps = np.reshape(self.kps, [-1, 3])

        split_filename = os.path.join(self.stereobj_root, 'split', self.split + '_' + self.cls_type + '.json')
        with open(split_filename, 'r') as f:
            filename_dict = json.load(f)

        self.filenames = []
        for subdir in filename_dict:
            for img_id in filename_dict[subdir]:
                self.filenames.append([subdir, img_id])
        self.filenames.sort()

        self._transforms = transforms
        self.num_kp = args.num_kp

    def load_cam_params(self):
        cam_param_filename = os.path.join(self.args.data, 'camera.json')
        with open(cam_param_filename, 'r') as f:
            cam_param = json.load(f)

        self.proj_matrix_l = np.array(cam_param['left']['P'])
        self.proj_matrix_r = np.array(cam_param['right']['P'])

        self.baseline = abs(self.proj_matrix_r[0, -1] / self.proj_matrix_r[0, 0])

    def read_data(self, img_id):

        path = os.path.join(self.stereobj_data_root, \
                img_id[0], img_id[1] + '.jpg')
        inp = Image.open(path)
        inp = inp.resize((2 * self.width, self.height))
        inp = np.asarray(inp)
        inp_l = inp[:, :self.width]
        ##### whether to read and process both left and right stereo images
        if self.lr:
            inp_r = inp[:, self.width:]

        if self.split != 'test':
            path = os.path.join(self.stereobj_data_root, \
                    img_id[0], img_id[1] + '_rt_label.json')
            with open(path, 'r') as f:
                rt_data = json.load(f)
            rt = None
            for obj in rt_data['class']:
                if rt_data['class'][obj] == self.cls_type:
                    rt = rt_data['rt'][obj]
                    break
            assert(rt is not None)
            R = np.array(rt['R'])
            t = np.array(rt['t'])
            cam_mat = self.proj_matrix_l[:, :-1]

            kps = np.dot(self.kps, R.T) + t
            kps_2d, _ = cv2.projectPoints(objectPoints=kps, \
                    rvec=np.zeros(shape=[3]), tvec=np.zeros(shape=[3]), \
                    cameraMatrix=cam_mat, distCoeffs=None)
            kps_2d[:, :, 0] = kps_2d[:, :, 0] / 1440 * self.width
            kps_2d[:, :, 1] = kps_2d[:, :, 1] / 1440 * self.height
            kps_2d = kps_2d[:, 0]
            kps_2d = kps_2d[:self.num_kp]
            kps_2d_l = np.copy(kps_2d)

            if self.lr:
                kps = np.dot(self.kps, R.T) + t + np.array([-self.baseline, 0, 0])
                kps_2d, _ = cv2.projectPoints(objectPoints=kps, \
                        rvec=np.zeros(shape=[3]), tvec=np.zeros(shape=[3]), \
                        cameraMatrix=cam_mat, distCoeffs=None)
                kps_2d[:, :, 0] = kps_2d[:, :, 0] / 1440 * self.width
                kps_2d[:, :, 1] = kps_2d[:, :, 1] / 1440 * self.height
                kps_2d = kps_2d[:, 0]
                kps_2d = kps_2d[:self.num_kp]
                kps_2d_r = np.copy(kps_2d)

            path = os.path.join(self.stereobj_data_root, \
                    img_id[0], img_id[1] + '_mask_label.npz')
            obj_mask = np.load(path)['masks'].item()
            ##### decode instance mask
            mask = np.zeros([1440, 1440], dtype='bool')

            mask_in_bbox = obj_mask['left'][obj]['mask']
            x_min = obj_mask['left'][obj]['x_min']
            x_max = obj_mask['left'][obj]['x_max']
            y_min = obj_mask['left'][obj]['y_min']
            y_max = obj_mask['left'][obj]['y_max']

            if x_min is not None:
                mask[y_min:(y_max+1), x_min:(x_max+1)] = mask_in_bbox
            mask = resize_binary_map(mask, (self.width, self.height))
            mask = mask.astype('uint8')
        else:
            kps_2d_l, kps_2d_r, mask, R, t = [], [], [], [], []

        if self.lr:
            return inp_l, inp_r, kps_2d_l, kps_2d_r, mask, R, t
        else:
            return inp_l, kps_2d_l, mask, R, t

    def __getitem__(self, index_tuple):
        if self.lr:
            return self.get_item_lr(index_tuple)
        else:
            return self.get_item_l(index_tuple)

    def get_item_l(self, index_tuple):
        # index, height, width = index_tuple
        index = index_tuple
        img_id = self.filenames[index]

        inp, kpt_2d, mask, R_gt, t_gt = self.read_data(img_id)

        view = False
        if view:
            import matplotlib.pyplot as plt
            plt.imshow(inp_l / 255.)
            plt.plot(kpt_2d_l[:, 0], kpt_2d_l[:, 1], 'ro')
            plt.figure()
            plt.imshow(inp_r / 255.)
            plt.plot(kpt_2d_r[:, 0], kpt_2d_r[:, 1], 'ro')
            plt.figure()
            plt.imshow(mask)
            plt.show()
            exit()

        if self.split != 'test':
            pose_gt = np.concatenate([R_gt, np.expand_dims(t_gt, -1)], axis=-1)
            if self._transforms is not None:
                inp, kpt_2d, mask, K = self._transforms(inp, kpt_2d, mask, self.proj_matrix_l)
            mask.astype(np.uint8)

            prob = data_utils.compute_prob(mask, kpt_2d)
        else:
            pose_gt = []
            prob = []

        ret = {'inp': inp, 'mask': mask, 'prob': prob, \
               'uv': kpt_2d, 'img_id': img_id, 'meta': {}, \
               'kpt_3d': self.kps[:self.num_kp], 'baseline': self.baseline, \
               'K': self.proj_matrix_l[:, :-1], 'pose_gt': pose_gt}
        return ret

    def get_item_lr(self, index_tuple):
        index = index_tuple
        img_id = self.filenames[index]

        inp_l, inp_r, kpt_2d_l, kpt_2d_r, mask, R_gt, t_gt = self.read_data(img_id)

        view = False
        if view:
            import matplotlib.pyplot as plt
            plt.imshow(inp_l / 255.)
            plt.plot(kpt_2d_l[:, 0], kpt_2d_l[:, 1], 'ro')
            plt.figure()
            plt.imshow(inp_r / 255.)
            plt.plot(kpt_2d_r[:, 0], kpt_2d_r[:, 1], 'ro')
            plt.figure()
            plt.imshow(mask)
            plt.show()
            exit()

        if self._transforms is not None:
            inp_l, kpt_2d_l, mask, K = self._transforms(inp_l, kpt_2d_l, mask, self.proj_matrix_l)
            inp_r, kpt_2d_r, mask, K = self._transforms(inp_r, kpt_2d_r, mask, self.proj_matrix_r)
            mask.astype(np.uint8)

        if self.split != 'test':
            pose_gt = np.concatenate([R_gt, np.expand_dims(t_gt, -1)], axis=-1)
            prob = data_utils.compute_prob(mask, kpt_2d_l)
        else:
            pose_gt = []
            prob = []

        ret = {'inp_l': inp_l, 'inp_r': inp_r, 'mask': mask, 'prob': prob, \
               'uv_l': kpt_2d_l, 'uv_r': kpt_2d_r, 'img_id': img_id, 'meta': {}, \
               'kpt_3d': self.kps[:self.num_kp], 'baseline': self.baseline, \
               'K': self.proj_matrix_l[:, :-1], 'pose_gt': pose_gt}
        return ret

    def __len__(self):
        return len(self.filenames)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((self.num_kp, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.rotate_min, self.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.overlap_ratio,
                                                         self.resize_ratio_min,
                                                         self.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask

if __name__ == '__main__':
    from transforms import make_transforms
    import argparse
    import matplotlib.pyplot as plt

    ##### rgb and mask renderer of meshes
    # please first go to `rgb_and_mask_renderer` and install the renderer package
    # by `python setup.py install`, then import it here
    import renderer as pytorch_renderer
    ##### rgb and mask renderer of meshes

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_kp', type=int, default=16, help='Number of Keypoints [default: 1024]')
    parser.add_argument('--image_width', type=int, default=1080, help='Image width [default: 768]')
    parser.add_argument('--image_height', type=int, default=1080, help='Image height [default: 768]')
    parser.add_argument('--data', default='/mnt/nas/xyl/stereobj_1m/images_annotations/', help='Data path to images_annotations/ directory [default: /mnt/nas/xyl/stereobj_1m/images_annotations/ ]')
    parser.add_argument('--split', default='train', help='Dataset split [default: train]')
    parser.add_argument('--cls_type', default='blade_razor', help='Object class of interest [default: ]')
    args = parser.parse_args()


    transforms = make_transforms(True)

    dataset = Dataset(args, transforms=transforms)
    data = dataset[600]
    print('-------------------')
    print('Keys in the dict:')
    print(list(data.keys()))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp = data['inp']
    mask = data['mask']
    prob = data['prob']
    uv = data['uv']
    img_id = data['img_id']
    rt = data['pose_gt']

    ##### viewing the keypoints
    inp = inp * std + mean

    print('-------------------')
    print('Viewing projected keypoints of:')
    print(img_id[0], img_id[1], args.cls_type)

    plt.imshow(inp)
    plt.scatter(uv[:, 0], uv[:, 1])
    plt.title(img_id[0] + ' ' + img_id[1] + ' ' + args.cls_type)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kp.png'))
    plt.show()
    ##### viewing the keypoints

    ##### read camera calibration parameters
    ### the camera calibration parameters are for 1440x1440 images
    ### for other resolutions, it needs to be scaled
    cam_idx = 3
    with open(os.path.join(dataset.args.data, 'camera.json'), 'r') as f:
        camera_param = json.load(f)

    baseline = abs(np.array(camera_param['right']['P'])[0, -1] / np.array(camera_param['right']['P'])[0, 0]) # baseline length in terms of meters
    K = np.array(camera_param['left']['P'])[:, :-1]
    K = torch.Tensor(K).cuda()
    K = K[None, :, :]
    ##### read camera calibration parameters

    ##### read the mesh file of the object
    vertices, faces = pytorch_renderer.load_obj(os.path.join(dataset.args.data, 'objects', dataset.args.cls_type + '.obj'), load_textures=False)
    vertices = vertices[None, :, :]  # [bs, num_vertices, 3]
    faces = faces[None, :, :]  # [bs, num_faces, 3]

    # bounding box of the mesh
    bbox_x_max = torch.max(vertices[0, :, 0])
    bbox_x_min = torch.min(vertices[0, :, 0])
    bbox_y_max = torch.max(vertices[0, :, 1])
    bbox_y_min = torch.min(vertices[0, :, 1])
    bbox_z_max = torch.max(vertices[0, :, 2])
    bbox_z_min = torch.min(vertices[0, :, 2])

    # deal with pipette_100_1000 specially, only for norm_coords
    if 'pipette_100_1000' in args.cls_type:
        obj_filepath = os.path.join(os.path.dirname(obj), 'pipette_0.5_10' + '.obj')
        vertices_spec, _ = pytorch_renderer.load_obj(obj_filepath, load_textures=False)
        vertices_spec = vertices_spec[None, :, :]  # [bs, num_vertices, 3]
        bbox_x_max = torch.max(vertices_spec[0, :, 0])
        bbox_x_min = torch.min(vertices_spec[0, :, 0])
        bbox_y_max = torch.max(vertices_spec[0, :, 1])
        bbox_y_min = torch.min(vertices_spec[0, :, 1])
        bbox_z_max = torch.max(vertices_spec[0, :, 2])
        bbox_z_min = torch.min(vertices_spec[0, :, 2])

    # normalized coordinate features
    norm_coord_x = (vertices[:, :, 0] - bbox_x_min) / (bbox_x_max - bbox_x_min)
    norm_coord_y = (vertices[:, :, 1] - bbox_y_min) / (bbox_y_max - bbox_y_min)
    norm_coord_z = (vertices[:, :, 2] - bbox_z_min) / (bbox_z_max - bbox_z_min)
    features = torch.stack((norm_coord_x, norm_coord_y, norm_coord_z), -1)

    ##### read the mesh file of the object


    ##### render normalized coordinate images
    IMAGE_HEIGHT = 1440
    IMAGE_WIDTH = 1440
    renderer = pytorch_renderer.Renderer(image_height=IMAGE_HEIGHT,
                          image_width=IMAGE_WIDTH,
                          camera_mode='projection', render_outside=True)

    R_np = rt[:, :3]
    t_np = rt[:, -1]

    R = torch.Tensor(R_np).cuda()
    R = R[None, :, :]
    t = torch.Tensor(t_np).cuda()
    t = t[None, None, :]

    rgb, mask, depth_map = renderer(vertices, \
            faces, features, K, R, t)

    rgb = rgb[0, IMAGE_HEIGHT:2*IMAGE_HEIGHT,
              IMAGE_WIDTH:2*IMAGE_WIDTH]
    mask = mask[0, IMAGE_HEIGHT:2*IMAGE_HEIGHT,
              IMAGE_WIDTH:2*IMAGE_WIDTH]

    rgb = (rgb * 255).astype('uint8')
    mask = (mask * 255).astype('uint8')
    rgb = cv2.resize(rgb, (args.image_width, args.image_height))
    mask = cv2.resize(mask, (args.image_width, args.image_height))

    image = (inp[:, :, ::-1] * 255).astype('uint8')
    image = cv2.resize(image, (args.image_width, args.image_height))

    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png'), image)
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'norm_coord.png'), rgb)
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask.png'), mask)
    ##### render normalized coordinate images

