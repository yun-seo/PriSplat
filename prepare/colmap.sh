#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Set to 0 if you do not have a GPU.
USE_GPU=1
# Path to a directory `base/` with images in `base/images/`.
DATASET_PATH=$1
# Recommended CAMERA values: OPENCV for perspective, OPENCV_FISHEYE for fisheye.
CAMERA=${2:-OPENCV}


# Run COLMAP.

### Feature extraction

colmap feature_extractor \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.use_gpu "$USE_GPU"


### Feature matching

colmap exhaustive_matcher \
    --database_path "$DATASET_PATH"/database.db \
    --SiftMatching.use_gpu "$USE_GPU"

## Use if your scene has > 500 images
## Replace this path with your own local copy of the file.
## Download from: https://demuc.de/colmap/#download
# VOCABTREE_PATH=/usr/local/google/home/bmild/vocab_tree_flickr100K_words32K.bin
# colmap vocab_tree_matcher \
#     --database_path "$DATASET_PATH"/database.db \
#     --VocabTreeMatching.vocab_tree_path $VOCABTREE_PATH \
#     --SiftMatching.use_gpu "$USE_GPU"


### Bundle adjustment

# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
mkdir -p "$DATASET_PATH"/sparse
colmap mapper \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --output_path "$DATASET_PATH"/sparse \
    --Mapper.ba_global_function_tolerance=0.000001


### Image undistortion

# Use this if you want to undistort your images into ideal pinhole intrinsics.
mkdir -p "$DATASET_PATH"/undistortion
colmap image_undistorter \
    --image_path "$DATASET_PATH"/images \
    --input_path "$DATASET_PATH"/sparse/0 \
    --output_path "$DATASET_PATH"/undistortion \
    --output_type COLMAP



# extra: move original data
mkdir -p "$DATASET_PATH"/raw && \
mv "$DATASET_PATH"/{images,sparse} "$DATASET_PATH"/raw/

# extra: move undistortion data
mv -f "$DATASET_PATH"/undistortion/images "$DATASET_PATH"/

# extra: reorganize undistortion data
mkdir -p "$DATASET_PATH"/sparse && \
mv "$DATASET_PATH"/undistortion/sparse "$DATASET_PATH"/sparse/0

# extra: remove empty folder 
rmdir "$DATASET_PATH"/undistortion 2>/dev/null || true