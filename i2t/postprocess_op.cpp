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

#include "include/postprocess_op.h"
#include "include/clipper.h"

namespace PaddleOCR {

    void DBPostProcessor::Init(double thresh, double box_thresh, double unclip_ratio,
        std::string score_mode, bool use_dilation) {
        this->det_db_thresh_ = thresh;
        this->det_db_box_thresh_ = box_thresh;
        this->det_db_unclip_ratio_ = unclip_ratio;
        this->det_db_score_mode_ = score_mode;
        this->use_dilation_ = use_dilation;
    }

    void DBPostProcessor::Run(const std::vector<float>& out_data,
        const std::vector<int>& output_shape,
        std::vector<std::vector<std::vector<int>>>& boxes,
        float ratio_h, float ratio_w,
        int src_h, int src_w,
        float unclip_ratio) {
        int n2 = output_shape[2];
        int n3 = output_shape[3];

        cv::Mat pred_map = cv::Mat(n2, n3, CV_32F, (float*)out_data.data());
        cv::Mat bitmap;

        cv::threshold(pred_map, bitmap, this->det_db_thresh_, 1.0, cv::THRESH_BINARY);

        cv::Mat bitmap_u8;
        bitmap.convertTo(bitmap_u8, CV_8UC1, 255);

        boxes = BoxesFromBitmap(pred_map, bitmap_u8, this->det_db_box_thresh_,
            unclip_ratio, this->det_db_score_mode_);

        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].size(); j++) {
                boxes[i][j][0] = int(boxes[i][j][0] / ratio_w);
                boxes[i][j][1] = int(boxes[i][j][1] / ratio_h);
                boxes[i][j][0] = std::min(std::max(boxes[i][j][0], 0), src_w - 1);
                boxes[i][j][1] = std::min(std::max(boxes[i][j][1], 0), src_h - 1);
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> DBPostProcessor::BoxesFromBitmap(
        const cv::Mat& pred, const cv::Mat& bitmap,
        const float& box_thresh, const float& det_db_unclip_ratio,
        const std::string& score_mode) {

        std::vector<std::vector<std::vector<int>>> boxes;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() <= 2) continue;
            float min_side = 3.0;

            cv::RotatedRect box = cv::minAreaRect(contours[i]);
            if (box.size.width < min_side || box.size.height < min_side) continue;

            cv::Point2f points[4];
            box.points(points);
            std::vector<std::vector<int>> int_box(4, std::vector<int>(2));
            for (int k = 0; k < 4; k++) {
                int_box[k][0] = (int)points[k].x;
                int_box[k][1] = (int)points[k].y;
            }

            float score = BoxScoreFast(int_box, pred);
            if (score < box_thresh) continue;

            ClipperLib::Path subject;
            for (const auto& pt : contours[i]) {
                subject.push_back(ClipperLib::IntPoint(pt.x, pt.y));
            }

            double area = cv::contourArea(contours[i]);
            double perimeter = cv::arcLength(contours[i], true);
            double distance = area * det_db_unclip_ratio / perimeter;

            ClipperLib::ClipperOffset offset;
            offset.AddPath(subject, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
            ClipperLib::Paths solution;
            offset.Execute(solution, distance);

            if (solution.empty()) continue;

            std::vector<std::vector<int>> result_box;
            for (const auto& pt : solution[0]) {
                result_box.push_back({ (int)pt.X, (int)pt.Y });
            }

            std::vector<cv::Point> cnt;
            for (auto& p : result_box) cnt.push_back(cv::Point(p[0], p[1]));
            cv::RotatedRect res_rect = cv::minAreaRect(cnt);

            cv::Point2f res_pts[4];
            res_rect.points(res_pts);

            std::vector<std::vector<int>> final_box(4, std::vector<int>(2));
            for (int k = 0; k < 4; k++) {
                final_box[k][0] = (int)res_pts[k].x;
                final_box[k][1] = (int)res_pts[k].y;
            }
            boxes.push_back(final_box);
        }
        return boxes;
    }

    float DBPostProcessor::BoxScoreFast(const std::vector<std::vector<int>>& box_array, const cv::Mat& pred) {
        int width = pred.cols;
        int height = pred.rows;
        int xmin = width, xmax = 0, ymin = height, ymax = 0;

        for (auto& pt : box_array) {
            xmin = std::min(xmin, pt[0]);
            xmax = std::max(xmax, pt[0]);
            ymin = std::min(ymin, pt[1]);
            ymax = std::max(ymax, pt[1]);
        }
        xmin = std::max(xmin, 0); xmax = std::min(xmax, width - 1);
        ymin = std::max(ymin, 0); ymax = std::min(ymax, height - 1);

        if (xmax - xmin <= 0 || ymax - ymin <= 0) return 0.0;

        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        std::vector<cv::Point> root_points;
        for (auto& pt : box_array) {
            root_points.push_back(cv::Point(pt[0] - xmin, pt[1] - ymin));
        }
        std::vector<std::vector<cv::Point>> pts = { root_points };
        cv::fillPoly(mask, pts, 1);

        cv::Mat croppedImg = pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        cv::Scalar mean = cv::mean(croppedImg, mask);
        return (float)mean[0];
    }

    void CTCLabelDecode::Run(const std::vector<float>& out_data,
        const std::vector<int>& output_shape,
        const std::vector<std::string>& label_list,
        std::string& text, float& score,
        int img_w, int img_h) {

        int time_step = output_shape[1];
        int num_class = output_shape[2];

        text = "";
        score = 0.0f;

        float space_factor = 0.8f;
        float calculated_thresh = ((float)img_h / (float)img_w) * (float)time_step * space_factor;
        int dynamic_threshold = std::max(2, (int)std::round(calculated_thresh));

        int count = 0;
        int pre_index = -1;
        int blank_consecutive_count = 0;

        std::vector<int> max_indices(time_step);
        std::vector<float> max_values(time_step);
        for (int i = 0; i < time_step; i++) {
            float max_value = -10000.0f;
            int max_index = 0;
            for (int j = 0; j < num_class; j++) {
                size_t idx = static_cast<size_t>(i) * num_class + j;
                if (idx >= out_data.size()) continue;
                float value = out_data[idx];
                if (value > max_value) { max_value = value; max_index = j; }
            }
            max_indices[i] = max_index;
            max_values[i] = max_value;
        }

        for (int i = 0; i < time_step; i++) {
            int max_index = max_indices[i];
            float max_value = max_values[i];

            if (max_index == 0) {
                blank_consecutive_count++;
            }
            else {
                // 계산된 Threshold보다 공백이 길면 띄어쓰기
                if (blank_consecutive_count > dynamic_threshold && text.length() > 0) {
                    text += " ";
                }
                blank_consecutive_count = 0;
            }

            if (max_index > 0 && max_index < label_list.size() && max_index != pre_index) {
                text += label_list[max_index];
                score += max_value;
                count++;
            }
            pre_index = max_index;
        }

        if (count > 0) {
            score /= count;
        }
    }
} // namespace PaddleOCR