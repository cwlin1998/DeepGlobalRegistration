import os
import glob
import scipy.io as scio
from PIL import Image

from dataloader.base_loader import *
from dataloader.transforms import *

from util.pointcloud import get_matching_indices, make_open3d_point_cloud

class YCBVideoPairDataset(PairDataset):
  '''
  Train dataset
  '''
  DATA_FILES = {
      'train': './dataloader/split/train_ycbvideo.txt',
      'val': './dataloader/split/val_ycbvideo.txt',
      'test': './dataloader/split/test_ycbvideo.txt'
  }

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.ycb_video_dir
    self.use_xyz_feature = config.use_xyz_feature
    logging.info(f"Loading the subset {phase} from {root}")

    classes = open(os.path.join(root, 'image_sets', 'classes.txt')).read().split()
    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      n_frames = len(os.listdir(os.path.join(root, 'data', name))) // 5
      for frame_idx in range(n_frames):
        frame_str = str(frame_idx+1).zfill(6)
        meta = scio.loadmat(os.path.join(root, 'data', name, frame_str+'-meta.mat'))
        scene_depth_file = os.path.join(root, 'data', name, frame_str+'-depth.png')
        scene_color_file = os.path.join(root, 'data', name, frame_str+'-color.png')
        scene_factor_depth = meta['factor_depth'].astype(np.float32)[0][0]
        scene_intrinsic_matrix = meta['intrinsic_matrix']
        for obj_idx, cls_idx in enumerate(meta['cls_indexes'].reshape(-1)):
          obj_file = os.path.join(root, 'models', classes[cls_idx-1], 'points.xyz')
          obj_pose = meta['poses'][:, :, obj_idx]
          self.files.append([scene_depth_file, scene_color_file, scene_factor_depth, scene_intrinsic_matrix, obj_file, obj_pose])


  def depth_2_pcd(self, depth, factor, K):
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    if len(depth.shape) > 2:
      depth = depth[:, :, 0]
    mask_depth = depth > 1e-6
    select = mask_depth.flatten().nonzero()[0].astype(np.uint32)
    if len(select) < 1:
      return None

    depth_masked = depth.flatten()[select][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[select][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[select][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / factor
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    pcd = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcd, select


  def __getitem__(self, idx):
    scene_depth_file, scene_color_file, \
    scene_factor_depth, scene_intrinsic_matrix, \
    obj_file, obj_pose = self.files[idx]

    depth = np.array(Image.open(scene_depth_file))
    color = np.array(Image.open(scene_color_file))

    xyz0, select = self.depth_2_pcd(depth, scene_factor_depth, scene_intrinsic_matrix)
    color0 = color.reshape(-1, 3)[select] / 255

    xyz1 = np.array([np.array(xyz.split(), dtype=np.float32) for xyz in open(obj_file).read().split('\n')[:-1]])
    xyz1 = self.apply_transform(xyz1, obj_pose)

    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Select features and points using the returned voxelized indices
    # pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    # pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    # xyz as feats
    if self.use_xyz_feature:
      feats_train0.append(unique_xyz0_th - unique_xyz0_th.mean(0))
      feats_train1.append(unique_xyz1_th - unique_xyz1_th.mean(0))
    else:
      feats_train0.append(torch.ones((npts0, 1)))
      feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    extra_package = {'idx': idx}

    return (unique_xyz0_th.float(),
            unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
            feats1.float(), matches, trans, extra_package)
