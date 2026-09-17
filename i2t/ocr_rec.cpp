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

#include "include/ocr_rec.h"
#include "include/utility.h"
#include <numeric>

namespace PaddleOCR {

    CRNNRecognizer::CRNNRecognizer(const std::string& model_dir, const bool& use_gpu,
        const int& gpu_id, const int& gpu_mem,
        const int& cpu_threads, const bool& enable_mkldnn,
        const std::string& label_path,
        const bool& use_tensorrt, const int& rec_img_h,
        const int& rec_img_w,
        const bool& use_space_char) {
        this->use_gpu_ = use_gpu;
        this->gpu_id_ = gpu_id;
        this->gpu_mem_ = gpu_mem;
        this->cpu_threads_ = cpu_threads;
        this->enable_mkldnn_ = enable_mkldnn;
        this->use_tensorrt_ = use_tensorrt;
        this->rec_img_h_ = rec_img_h;
        this->rec_img_w_ = rec_img_w;
        this->use_space_char_ = use_space_char;

        this->label_list_ = Utility::ReadDict(label_path);
        this->label_list_.insert(this->label_list_.begin(), "!");
        if (use_space_char) {
            this->label_list_.push_back(" ");
        }

        LoadModel(model_dir);
    }

    void CRNNRecognizer::LoadModel(const std::string& model_dir) {
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

    void CRNNRecognizer::Run(cv::Mat& img, std::vector<std::vector<std::vector<int>>>& boxes,
        std::vector<std::string>& texts, std::vector<float>& rec_scores) {
        texts.clear();
        rec_scores.clear();

        std::vector<std::vector<std::vector<int>>> sorted_boxes = Utility::SortedBoxes(boxes);

        for (int i = 0; i < sorted_boxes.size(); i++) {
            cv::Mat crop_img = Utility::GetRotateCropImage(img, sorted_boxes[i]);
            float wh_ratio = float(crop_img.cols) / float(crop_img.rows);
            cv::Mat resize_img;

            this->resize_op_.Run(crop_img, resize_img, wh_ratio, this->use_tensorrt_,
                this->rec_img_h_, this->rec_img_w_);
            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, this->is_scale_);
            std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
            this->permute_op_.Run(&resize_img, input.data());

            // [중요] v3.0 텐서 입력 방식
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

            std::string text;
            float score;
            PaddleOCR::CTCLabelDecode ctc_decoder;
            ctc_decoder.Run(out_data, output_shape, this->label_list_, text, score, crop_img.cols, crop_img.rows);

            texts.push_back(text);
            rec_scores.push_back(score);
        }
    }

} // namespace PaddleOCR