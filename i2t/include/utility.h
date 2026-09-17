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
#include <vector>
#include <string>
#include <opencv2/core.hpp>

class Utility {
public:
    static std::vector<std::string> ReadDict(const std::string& path);
    static std::vector<std::vector<std::vector<int>>> SortedBoxes(
        const std::vector<std::vector<std::vector<int>>>& boxes);
    static cv::Mat GetRotateCropImage(const cv::Mat& img,
        const std::vector<std::vector<int>>& box);
    static cv::Mat VisualizeBboxes(const cv::Mat& srcimg,
        const std::vector<std::vector<std::vector<int>>>& boxes);
private:
    static bool XsortInt(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
    static bool YsortInt(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
};