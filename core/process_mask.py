import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time

import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse

import tensorflow as tf

# code borrowed from: https://github.com/senguptaumd/Background-Matting/blob/master/test_segmentation_deeplab.py

## setup ####################
def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
          colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
    label: A 2D array with integer type, storing the segmentation label.
    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]



LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = _MODEL_URLS[MODEL_NAME]



class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        #"""Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
          image: A PIL.Image object, raw input image.
        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def process_masks(img_paths, save_paths, model_dir='deeplab_model'):

    # download model if not already have it
    if not os.path.exists(model_dir):
      tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
      print('downloading model to %s, this might take a while...' % download_path)
      urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                     download_path)
      print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')

    # process images
    for i in range(0, len(img_paths)):
        start_time = time.time()
        if i % 500 == 0:
            print(f"{i+1}/{len(img_paths)}")


        start_read = time.time()
        image = Image.open(img_paths[i])
        start_read = time.time()
        res_im,seg=MODEL.run(image)

        seg=cv2.resize(seg.astype(np.uint8),image.size)

        mask_sel=(seg==15).astype(np.float32)

        save_dir = os.path.dirname(save_paths[i])
        os.makedirs(save_dir, exist_ok=True)

        # dilate the boundary a bit because as the mask is not accurate
        kernel = np.ones((3, 3))
        mask_sel = cv2.dilate(mask_sel, kernel=kernel, iterations=1)

        start_read = time.time()
        cv2.imwrite(save_paths[i],(255*mask_sel).astype(np.uint8))
    print("finish mask processing.")

def process_bbox_masks(img_paths, save_paths, bboxes, model_dir='deeplab_model', mul=1.0):

    if not os.path.exists(model_dir):
      tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
      print('downloading model to %s, this might take a while...' % download_path)
      urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                     download_path)
      print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')

    # process images
    for i in range(0, len(img_paths)):
        if i % 500 == 0:
            print(f"{i+1}/{len(img_paths)}")


        start_read = time.time()
        image = Image.open(img_paths[i])
        W, H = image.size

        # crop by bounding box
        cx, cy, box_len = bboxes[i]
        cx, cy = int(cx), int(cy)
        box_len = int(box_len * 0.5 * mul)
        left = max(cx - box_len, 0)
        top = max(cy - box_len, 0)
        right = min(cx + box_len, W)
        bot = min(cy + box_len, H)

        cropped = image.crop((left, top, right, bot))

        res_im,seg=MODEL.run(cropped)

        seg=cv2.resize(seg.astype(np.uint8),cropped.size)
        mask_sel= ((seg==15).astype(np.uint8))
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[top:bot, left:right] = mask_sel
        # put the cropped part back to the original image

        save_dir = os.path.dirname(save_paths[i])
        os.makedirs(save_dir, exist_ok=True)

        # dilate the boundary a bit because as the mask is not accurate
        kernel = np.ones((3, 3))
        mask = cv2.dilate(mask, kernel=kernel, iterations=1)[..., None]
        cv2.imwrite(save_paths[i],(255*mask).astype(np.uint8))

    print("finish mask processing.")



if __name__ == "__main__":
    import deepdish as dd
    import imageio
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-b", "--base_path", type=str,
                        #default="data/h36m/h36m_full",
                        default='data/h36m/',
                        help='base directory')
    parser.add_argument("-t", "--type", type=str, default='h36m',
                        help='type of data to process')
    parser.add_argument("-c", "--camera_id", type=int, default=None,
                        help='camera to extract')
    parser.add_argument("-s", "--subject", type=str, default="S9",
                        help='subject to extract')
    parser.add_argument("-r", "--res", type=float, default=1.0,
                        help='mask resolution')
    parser.add_argument("--h5_path", type=str, default=None,
                        help='path to a .h5 file, mainly for MonPerfCap')
    args = parser.parse_args()
    base_path = args.base_path #"data/h36m/h36m_full"

    #if args.h5_path is None:
    if args.type == 'h36m':
        subject = args.subject # "S9"
        camera_id = args.camera_id # -1

        cameras = ["54138969", "55011271", "58860488", "60457274"]
        camera = None

        if camera_id is not None:
            camera = cameras[camera_id]
            if subject != 'S1':
                h5_name = os.path.join(base_path, f"{subject}-camera=[{camera}]-subsample=5.h5")
            else:
                h5_name = os.path.join(base_path, f"{subject}-camera=[{camera}]-subsample=1.h5")
        else:
            h5_name = os.path.join(base_path, f"{subject}_SPIN_rect_output-maxmin.h5")
        print(h5_name)
        img_paths = dd.io.load(h5_name, "/img_path")
        bboxes = dd.io.load(h5_name, "/bbox_params")
        img_paths = [os.path.join(base_path, img_path) for img_path in img_paths]
        mask_paths = [img_path.replace(f"{subject}", f"{subject}m_") for img_path in img_paths]
        #process_masks(img_paths, mask_paths, res=args.res)
        #process_masks(img_paths, mask_paths)
        process_bbox_masks(img_paths, mask_paths, bboxes, mul=1.1)
        """
        cameras = ["54138969", "55011271", "58860488", "60457274"]

        base_path = args.base_path
        subject = args.subject
        h5_name = os.path.join(base_path, f"{subject}_processed.h5")
        img_paths = dd.io.load(h5_name, "/img_path")
        img_paths = [os.path.join(base_path, img_path) for img_path in img_paths]
        mask_paths = [img_path.replace(f"{subject}", f"{subject}m") for img_path in img_paths]
        process_masks(img_paths, mask_paths, res=args.res)

        """
    elif args.type == 'perfcap':
        from load_perfcap import read_spin_data
        processed_est = read_spin_data(args.h5_path)

        img_paths = processed_est["img_path"]
        img_paths = [os.path.join(args.base_path, img_path) for img_path in img_paths]
        save_paths = [img_path.replace("/images/", "/masks/") for img_path in img_paths]
        process_bbox_masks(img_paths, save_paths, processed_est["bboxes"])
    elif args.type == '3dhp':
        from load_3dhp import read_3dhp_spin_data
        subject = args.subject # "S9"
        processed_est = read_3dhp_spin_data(args.h5_path, subject=subject)

        img_paths = processed_est["img_path"]
        img_paths = [os.path.join(args.base_path, img_path) for img_path in img_paths]
        save_paths = [img_path.replace("/imageSequence/", "/masks/") for img_path in img_paths]
        process_bbox_masks(img_paths, save_paths, processed_est["bboxes"])
