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

#include "include/preprocess_op.h"

namespace PaddleOCR {

    // 1. Permute 구현
    void Permute::Run(const cv::Mat* im, float* data) {
        int rh = im->rows;
        int rw = im->cols;
        int rc = im->channels();
        for (int i = 0; i < rc; ++i) {
            for (int j = 0; j < rh; ++j) {
                for (int k = 0; k < rw; ++k) {
                    data[i * rh * rw + j * rw + k] =
                        (float)im->at<cv::Vec3f>(j, k)[i];
                }
            }
        }
    }

    // 2. Normalize 구현
    void Normalize::Run(cv::Mat* im, const std::vector<float>& mean,
        const std::vector<float>& scale, const bool is_scale) {
        double e = 1.0;
        if (is_scale) {
            e /= 255.0;
        }
        (*im).convertTo(*im, CV_32FC3, e);
        std::vector<cv::Mat> bgr_channels(3);
        cv::split(*im, bgr_channels);
        for (auto i = 0; i < bgr_channels.size(); i++) {
            bgr_channels[i] = (bgr_channels[i] - mean[i]) * scale[i];
        }
        cv::merge(bgr_channels, *im);
    }

    // 3. ResizeImgType0 구현 (Detection 용)
    void ResizeImgType0::Run(const cv::Mat& img, cv::Mat& resize_img,
        const std::string& limit_type, int limit_side_len,
        float& ratio_h, float& ratio_w, bool use_tensorrt) {
        int w = img.cols;
        int h = img.rows;
        float ratio = 1.f;

        if (limit_type == "min") {
            int min_wh = std::min(h, w);
            if (min_wh < limit_side_len) {
                if (h < w) {
                    ratio = float(limit_side_len) / float(h);
                }
                else {
                    ratio = float(limit_side_len) / float(w);
                }
            }
        }
        else {
            int max_wh = std::max(h, w);
            if (max_wh > limit_side_len) {
                if (h > w) {
                    ratio = float(limit_side_len) / float(h);
                }
                else {
                    ratio = float(limit_side_len) / float(w);
                }
            }
        }

        int resize_h = int(h * ratio);
        int resize_w = int(w * ratio);

        resize_h = std::max(int(round(resize_h / 32.0) * 32), 32);
        resize_w = std::max(int(round(resize_w / 32.0) * 32), 32);

        cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

        ratio_h = float(resize_h) / float(h);
        ratio_w = float(resize_w) / float(w);
    }

    // 4. CrnnResizeImg 구현 (Recognition 용) - [이 부분이 없어서 에러가 난 것입니다!]
    void CrnnResizeImg::Run(const cv::Mat& img, cv::Mat& resize_img,
        float wh_ratio, bool use_tensorrt,
        int rec_img_h, int rec_img_w) {

        // 높이는 고정(rec_img_h), 너비는 비율에 따라 계산
        int imgW = int(rec_img_h * wh_ratio);

        float ratio = (float)img.cols / (float)img.rows;
        int resize_w;

        if (use_tensorrt) {
            resize_w = rec_img_w;
        }
        else {
            resize_w = (int)(rec_img_h * ratio);
            if (resize_w > rec_img_w) resize_w = rec_img_w;
        }

        if (resize_w < 10) resize_w = 10; // 최소 너비 보정

        cv::resize(img, resize_img, cv::Size(resize_w, rec_img_h));
    }

} // namespace PaddleOCR