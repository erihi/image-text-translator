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

// [수정] string 헤더 필수 포함
#include <string> 
#include <vector>
#include <iostream>

// OpenCV 헤더
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace PaddleOCR {

    // 1. Normalize (공통)
    class Normalize {
    public:
        virtual void Run(cv::Mat* im, const std::vector<float>& mean,
            const std::vector<float>& scale, const bool is_scale = true);
    };

    // 2. Permute (공통)
    class Permute {
    public:
        virtual void Run(const cv::Mat* im, float* data);
    };

    // 3. ResizeImgType0 (Detection 용)
    class ResizeImgType0 {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img,
            const std::string& limit_type, int limit_side_len,
            float& ratio_h, float& ratio_w, bool use_tensorrt);
    };

    // 4. CrnnResizeImg (Recognition 용)
    class CrnnResizeImg {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio,
            bool use_tensorrt, int rec_img_h, int rec_img_w);
    };

} // namespace PaddleOCR