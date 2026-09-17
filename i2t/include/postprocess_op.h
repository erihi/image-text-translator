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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <string>

namespace PaddleOCR {

    class DBPostProcessor {
    public:
        void Init(double thresh, double box_thresh, double unclip_ratio,
            std::string score_mode, bool use_dilation);

        // [중요] 튜닝된 ratio를 인자로 받도록 설계됨
        void Run(const std::vector<float>& out_data,
            const std::vector<int>& output_shape,
            std::vector<std::vector<std::vector<int>>>& boxes,
            float ratio_h, float ratio_w,
            int src_h, int src_w,
            float unclip_ratio);

        std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
            const cv::Mat& pred, const cv::Mat& bitmap,
            const float& box_thresh, const float& det_db_unclip_ratio,
            const std::string& score_mode);

        std::vector<std::vector<int>> FilterTagDetRes(
            std::vector<std::vector<int>> boxes, float ratio_h, float ratio_w,
            cv::Mat srcimg);

    private:
        float BoxScoreFast(const std::vector<std::vector<int>>& box_array, const cv::Mat& pred);
        float PolygonScoreAcc(const std::vector<cv::Point>& contour, const cv::Mat& pred);
        float UnclipScore(const std::vector<std::vector<int>>& box_array, const cv::Mat& pred);

        double det_db_thresh_ = 0.3;
        double det_db_box_thresh_ = 0.5;
        double det_db_unclip_ratio_ = 2.0;
        std::string det_db_score_mode_ = "slow";
        bool use_dilation_ = false;
    };

    // [중요] Recognition 결과 디코딩용 클래스
    class CTCLabelDecode {
    public:
        void Run(const std::vector<float>& out_data,
            const std::vector<int>& output_shape,
            const std::vector<std::string>& label_list,
            std::string& text, float& score, int img_w, int img_h);
    };

} // namespace PaddleOCR