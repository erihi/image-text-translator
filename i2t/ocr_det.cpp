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

#include "include/ocr_det.h"
#include <numeric>

namespace PaddleOCR {

    DBDetector::DBDetector(const std::string& model_dir, const bool& use_gpu,
        const int& gpu_id, const int& gpu_mem,
        const int& cpu_threads, const bool& enable_mkldnn,
        const std::string& limit_type,
        const int& limit_side_len, const double& det_db_thresh,
        const double& det_db_box_thresh,
        const double& det_db_unclip_ratio,
        const bool& use_dilation,
        const std::string& det_db_score_mode) {
        this->use_gpu_ = use_gpu;
        this->gpu_id_ = gpu_id;
        this->gpu_mem_ = gpu_mem;
        this->cpu_threads_ = cpu_threads;
        this->enable_mkldnn_ = enable_mkldnn;
        this->limit_type_ = limit_type;
        this->limit_side_len_ = limit_side_len;
        this->det_db_thresh_ = det_db_thresh;
        this->det_db_box_thresh_ = det_db_box_thresh;
        this->det_db_unclip_ratio_ = det_db_unclip_ratio;
        this->use_dilation_ = use_dilation;
        this->det_db_score_mode_ = det_db_score_mode;

        LoadModel(model_dir);
    }

    void DBDetector::LoadModel(const std::string& model_dir) {
        // [중요] v3.0 설정 방식
        paddle_infer::Config config;
        config.SetModel(model_dir + "/inference.pdmodel",
            model_dir + "/inference.pdiparams");

        if (this->use_gpu_) {
            config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
        }
        else {
            config.DisableGpu();
            if (this->enable_mkldnn_) config.EnableMKLDNN();
            config.SetCpuMathLibraryNumThreads(this->cpu_threads_);
        }

        config.SwitchUseFeedFetchOps(true);
        config.SwitchIrOptim(true);

        // [중요] v3.0 CreatePredictor
        this->predictor_ = paddle_infer::CreatePredictor(config);
    }

    void DBDetector::Run(cv::Mat& img, std::vector<std::vector<std::vector<int>>>& boxes, float ratio) {
        float ratio_h = 1.0; float ratio_w = 1.0;
        cv::Mat srcimg; cv::Mat resize_img;
        img.copyTo(srcimg);

        this->resize_op_.Run(img, resize_img, this->limit_type_,
            this->limit_side_len_, ratio_h, ratio_w, false);

        this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, this->is_scale_);

        std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
        this->permute_op_.Run(&resize_img, input.data());

        // [중요] v3.0 텐서 입력 방식 (GetInputHandle)
        auto input_names = this->predictor_->GetInputNames();
        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        input_t->Reshape({ 1, 3, resize_img.rows, resize_img.cols });
        input_t->CopyFromCpu(input.data());

        this->predictor_->Run();

        auto output_names = this->predictor_->GetOutputNames();
        auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
        std::vector<int> output_shape = output_t->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
        std::vector<float> out_data(out_num);
        output_t->CopyToCpu(out_data.data());

        this->post_processor_.Run(out_data, output_shape, boxes, ratio_h, ratio_w,
            srcimg.rows, srcimg.cols, ratio);
    }

} // namespace PaddleOCR