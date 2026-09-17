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

// 네임스페이스가 없다면 전역으로 사용
namespace PaddleOCR {

    struct StructurePredictResult {
        std::vector<float> box;
        std::string type;
        float confidence;
        std::string text;
    };

    // [중요] OCR 결과 담는 구조체
    struct OCRPredictResult {
        // 1. 박스 좌표 (4개의 점)
        std::vector<std::vector<int>> box;

        // 2. 인식된 텍스트
        std::string text;

        // 3. 신뢰도 점수 (반드시 float여야 함!)
        float score = -1.0;

        // 4. (선택) 분류 점수
        float cls_score = -1.0;
        int cls_label = -1;
    };

} // namespace PaddleOCR
