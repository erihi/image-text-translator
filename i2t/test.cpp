#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <utility>
#include <thread>

#include "DragDrop.h"

#include "include/args.h"
#include "include/ocr_det.h"
#include "include/ocr_rec.h"

using namespace std;

cv::Mat ImreadSafe(const string& path_utf8) {
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, &path_utf8[0], (int)path_utf8.size(), NULL, 0);
	std::wstring wpath(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, &path_utf8[0], (int)path_utf8.size(), &wpath[0], size_needed);

	FILE* f = nullptr;
	_wfopen_s(&f, wpath.c_str(), L"rb");
	if (!f) return cv::Mat();

	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uchar> buf(sz);
	fread(buf.data(), 1, sz, f);
	fclose(f);

	return cv::imdecode(buf, cv::IMREAD_COLOR);
}

// --- 전처리: 패딩 추가 ---
cv::Mat PreprocessImage(const std::string& image_path) {
	cv::Mat img = ImreadSafe(image_path);
	if (img.empty()) return cv::Mat();

	int padding = 50;
	cv::Mat padded;
	cv::copyMakeBorder(img, padded, padding, padding, padding, padding,
		cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	return padded;
}

// --- 난수 생성기 ---
std::mt19937& get_rng() {
	static random_device rd;
	static mt19937 gen(rd());
	return gen;
}

float get_random_float(float min, float max) {
	std::uniform_real_distribution<float> dis(min, max);
	return round(dis(get_rng()) * 100.0f) / 100.0f;
}

int get_random_int(int min, int max) {
	std::uniform_int_distribution<int> dis(min, max);
	return dis(get_rng());
}

int main(int argc, char** argv) {
	system("chcp 65001 > nul");
	HINSTANCE hInst = GetModuleHandle(NULL);
	DragDropWindow window(hInst);

	if (!window.Initialize(L"PaddleOCR - Drag and Drop", 600, 600)) {
		MessageBoxA(NULL, "Window Initialization Failed!", "Error", MB_ICONERROR);
		return -1;
	}

	window.RunMessageLoop();
	// 2. GUI 결과 데이터 추출
	string imagePath = g_droppedFilePath;
	int source_lang = stoi(g_lang);

	if (imagePath.empty()) {
		cout << "No file dropped. Exiting..." << endl;
		return 0;
	}
	cout << source_lang << endl;

	// 3. 선택된 언어에 따른 모델 경로 설정
	FLAGS_det_model_dir = "./inference/ch_PP-OCRv4_det_infer"; // Detection 모델은 공통

	if (source_lang == 1) { // 한국어
		FLAGS_rec_model_dir = "./inference/korean_PP-OCRv4_rec_infer";
		FLAGS_rec_char_dict_path = "./inference/korean_dict.txt";
	}
	else if (source_lang == 2) { // 영어
		FLAGS_rec_model_dir = "./inference/en_PP-OCRv4_rec_infer";
		FLAGS_rec_char_dict_path = "./inference/en_dict.txt";
	}
	else { // 중국어
		FLAGS_rec_model_dir = "./inference/ch_PP-OCRv4_rec_infer";
		FLAGS_rec_char_dict_path = "./inference/ppocr_keys_v1.txt";
	}

	FLAGS_rec_img_h = 48;
	FLAGS_rec_img_w = 2560;
	FLAGS_use_space_char = true;

	// 4. PaddleOCR 엔진 초기화
	PaddleOCR::DBDetector det(FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
		FLAGS_gpu_mem, FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_limit_type,
		FLAGS_limit_side_len, FLAGS_det_db_thresh, FLAGS_det_db_box_thresh,
		FLAGS_det_db_unclip_ratio, FLAGS_use_dilation, FLAGS_det_db_score_mode);

	PaddleOCR::CRNNRecognizer rec(FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
		FLAGS_gpu_mem, FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_rec_char_dict_path,
		FLAGS_use_tensorrt, FLAGS_rec_img_h, FLAGS_rec_img_w, FLAGS_use_space_char);

	// 5. 이미지 로드 및 분석 (튜닝 루프 시작)
	cv::Mat img_raw = ImreadSafe(imagePath);
	if (img_raw.empty()) { cout << L"이미지 로드 실패: " << imagePath << endl; return -1; }
	cv::Mat img_pre = PreprocessImage(imagePath);

	vector<cv::Mat> images = { img_raw, img_pre };
	int max_trials = 30, valid_trials = 0;
	float best_score = -1.0f, best_ratio = 2.0f;
	int best_img_idx = 0;
	vector<string> best_texts;
	set<pair<int, float>> tested_ratios;
	vector<float> fixed_candidates = { 1.6f, 2.0f, 2.4f, 2.8f };

	while (valid_trials < max_trials) {
		float current_ratio = 2.0f;
		int current_img_idx = 0;

		// 탐색 전략 (정해진 후보군 -> 이후 랜덤 탐색)
		if (valid_trials < 8) {
			current_ratio = fixed_candidates[valid_trials % 4];
			current_img_idx = (valid_trials < 4) ? 0 : 1;
		}
		else {
			current_img_idx = get_random_int(0, 1);
			current_ratio = get_random_float(1.2f, 3.5f);
		}

		if (tested_ratios.count({ current_img_idx, current_ratio })) {
			valid_trials++;
			continue;
		}

		vector<vector<vector<int>>> boxes;
		det.Run(images[current_img_idx], boxes, current_ratio);

		if (!boxes.empty()) {
			vector<string> texts;
			vector<float> rec_scores;
			rec.Run(images[current_img_idx], boxes, texts, rec_scores);

			float sum = 0;
			for (float s : rec_scores) sum += s;
			float avg_score = rec_scores.empty() ? 0 : sum / rec_scores.size();

			if (avg_score > best_score) {
				best_score = avg_score;
				best_ratio = current_ratio;
				best_texts = texts;
				best_img_idx = current_img_idx;
			}
			if (best_score > 0.99f) break;
		}
		tested_ratios.insert({ current_img_idx, current_ratio });
		valid_trials++;
	}

	// 6. 결과 출력
	for (const auto& txt : best_texts) {
		cout << txt << " ";
	}
	return 0;
}