from PIL import Image
import os
from os.path import isfile, join
import json
import sys
import image_embeddings
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path


MODE = 'val'           # train or val 

path_to_parquet = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_embed/tmp'.format(MODE)
path_to_embed = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_embed'.format(MODE)

pqs = [f for f in os.listdir(path_to_parquet) if isfile(join(path_to_parquet, f))] 

for i, p in enumerate(pqs):
    p_path = join(path_to_parquet, p)
    emb = pq.read_table(p_path).to_pandas()

    Path(path_to_embed).mkdir(parents=True, exist_ok=True)
    id_name = [{"id": k, "name": v.decode("utf-8")} for k, v in enumerate(list(emb["image_name"]))]
    json.dump(id_name, open(path_to_embed + "/id_name{}.json".format(i), "w"))

    emb = np.stack(emb["embedding"].to_numpy())
    np.save(open(path_to_embed + "/embedding{}.npy".format(i), "wb"), emb)
