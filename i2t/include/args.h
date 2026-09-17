// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTE: This file is adapted from PaddleOCR's official C++ inference
// deployment sample (deploy/cpp_infer) and modified for this project.

#pragma once
#include <string>

static bool FLAGS_use_gpu = false;
static int FLAGS_gpu_id = 0;
static int FLAGS_gpu_mem = 4000;
static int FLAGS_cpu_threads = 10;
static bool FLAGS_enable_mkldnn = true;
static bool FLAGS_use_tensorrt = false;
static std::string FLAGS_limit_type = "max";
static int FLAGS_limit_side_len = 2560;
static double FLAGS_det_db_thresh = 0.3;
static double FLAGS_det_db_box_thresh = 10;
static double FLAGS_det_db_unclip_ratio = 0.5;
static bool FLAGS_use_dilation = false;
static std::string FLAGS_det_db_score_mode = "slow";
static bool FLAGS_visualize = true;
static bool FLAGS_benchmark = false;
static bool FLAGS_use_space_char = true;

static std::string FLAGS_det_model_dir = "";
static std::string FLAGS_rec_model_dir = "";
static std::string FLAGS_rec_char_dict_path = "";

// Rec 이미지 크기 (C, H, W)
static std::string FLAGS_rec_image_shape = "3, 48, 320";
static int FLAGS_rec_batch_num = 6;
static int FLAGS_rec_img_h = 48;
static int FLAGS_rec_img_w = 320;