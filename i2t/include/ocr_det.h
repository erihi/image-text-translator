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
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <include/paddle_inference_api.h> // v3.0 헤더

#include "include/postprocess_op.h"
#include "include/preprocess_op.h"

namespace PaddleOCR {

    class DBDetector {
    public:
        explicit DBDetector(const std::string& model_dir, const bool& use_gpu,
            const int& gpu_id, const int& gpu_mem,
            const int& cpu_threads, const bool& enable_mkldnn,
            const std::string& limit_type,
            const int& limit_side_len, const double& det_db_thresh,
            const double& det_db_box_thresh,
            const double& det_db_unclip_ratio,
            const bool& use_dilation,
            const std::string& det_db_score_mode);

        void Run(cv::Mat& img, std::vector<std::vector<std::vector<int>>>& boxes, float ratio);

    private:
        void LoadModel(const std::string& model_dir);

        // [중요] v3.0은 paddle_infer 네임스페이스 사용
        std::shared_ptr<paddle_infer::Predictor> predictor_;

        bool use_gpu_;
        int gpu_id_;
        int gpu_mem_;
        int cpu_threads_;
        bool enable_mkldnn_;
        std::string limit_type_;
        int limit_side_len_;
        double det_db_thresh_;
        double det_db_box_thresh_;
        double det_db_unclip_ratio_;
        bool use_dilation_;
        std::string det_db_score_mode_;

        ResizeImgType0 resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;
        DBPostProcessor post_processor_;

        const std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
        const std::vector<float> scale_ = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
        const bool is_scale_ = true;
    };

} // namespace PaddleOCR