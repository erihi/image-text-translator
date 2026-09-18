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

#include "include/utility.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 1. ���� ���� �б�
std::vector<std::string> Utility::ReadDict(const std::string& path) {
    std::vector<std::string> dict;
    std::ifstream in(path);
    std::string line;
    if (in) {
        while (getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            dict.push_back(line);
        }
    }
    else {
        std::cerr << "Cannot open dictionary file: " << path << std::endl;
        exit(1);
    }
    return dict;
}

// 2. ���� �� �Լ�
bool Utility::XsortInt(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
    return a[0][0] < b[0][0];
}

bool Utility::YsortInt(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
    return a[0][1] < b[0][1];
}

// 3. �ڽ� ���� ����
std::vector<std::vector<std::vector<int>>> Utility::SortedBoxes(
    const std::vector<std::vector<std::vector<int>>>& boxes) {

    std::vector<std::vector<std::vector<int>>> sorted_boxes = boxes;

    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
        [](const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
            if (abs(a[0][1] - b[0][1]) < 10) {
                return a[0][0] < b[0][0];
            }
            return a[0][1] < b[0][1];
        });

    return sorted_boxes;
}

// 4. �̹��� ũ�� (Perspective Transform)
cv::Mat Utility::GetRotateCropImage(const cv::Mat& img,
    const std::vector<std::vector<int>>& box) {
    cv::Mat image;
    img.copyTo(image);

    cv::Point2f src_pts[4];
    for (int i = 0; i < 4; i++) {
        src_pts[i] = cv::Point2f(static_cast<float>(box[i][0]), static_cast<float>(box[i][1]));
    }

    float width1 = sqrt(pow(src_pts[0].x - src_pts[1].x, 2) + pow(src_pts[0].y - src_pts[1].y, 2));
    float width2 = sqrt(pow(src_pts[2].x - src_pts[3].x, 2) + pow(src_pts[2].y - src_pts[3].y, 2));
    float img_w = std::max(width1, width2);

    float height1 = sqrt(pow(src_pts[0].x - src_pts[3].x, 2) + pow(src_pts[0].y - src_pts[3].y, 2));
    float height2 = sqrt(pow(src_pts[1].x - src_pts[2].x, 2) + pow(src_pts[1].y - src_pts[2].y, 2));
    float img_h = std::max(height1, height2);

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0.f, 0.f);
    pts_std[1] = cv::Point2f(img_w, 0.f);
    pts_std[2] = cv::Point2f(img_w, img_h);
    pts_std[3] = cv::Point2f(0.f, img_h);

    cv::Mat M = cv::getPerspectiveTransform(src_pts, pts_std);
    cv::Mat img_crop;
    cv::warpPerspective(image, img_crop, M, cv::Size((int)img_w, (int)img_h),
        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    if (float(img_crop.rows) >= float(img_crop.cols) * 1.5) {
        cv::Mat srcClone = img_crop.clone();
        cv::transpose(srcClone, img_crop);
        cv::flip(img_crop, img_crop, 0);
    }

    return img_crop;
}

cv::Mat Utility::VisualizeBboxes(const cv::Mat& srcimg,
    const std::vector<std::vector<std::vector<int>>>& boxes) {
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (size_t i = 0; i < boxes.size(); i++) {
        std::vector<cv::Point> cnt;
        for (const auto& pt : boxes[i]) {
            cnt.push_back(cv::Point(pt[0], pt[1]));
        }
        cv::polylines(img_vis, cnt, true, cv::Scalar(0, 255, 0), 2);
    }
    return img_vis;
}