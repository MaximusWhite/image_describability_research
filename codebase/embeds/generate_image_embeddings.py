from PIL import Image
import os
from os.path import isfile, join
import json
import sys
import image_embeddings

MODE = 'val'      # train or val 

path_to_resized = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_resized'.format(MODE)
path_to_tf = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_tf'.format(MODE)
path_to_embed = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_embed/tmp'.format(MODE)

image_embeddings.inference.write_tfrecord(image_folder=path_to_resized,
                                          output_folder=path_to_tf,
                                          num_shards=10)

image_embeddings.inference.run_inference(tfrecords_folder=path_to_tf,
                                         output_folder=path_to_embed,
                                         batch_size=100)

