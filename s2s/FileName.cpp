#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <sstream>  
#include <algorithm> 
#include <windows.h>
#include <sentencepiece_processor.h>
#include <ctranslate2/translator.h>
#include <cctype>

using namespace std;

// 전역 변수 및 핸들
int g_selected_lang = 0;
int source_lang_id = 2;
string total_text;
string input_text;
HWND hResultStatic;

ctranslate2::Translator* g_translator = nullptr;
sentencepiece::SentencePieceProcessor* g_processor = nullptr;

// 문장 분리 함수
vector<string> SplitSentences(const string& text) {
    vector<string> sentences;
    string current_sentence;
    const int SAFE_LIMIT = 900;
    const int MAX_LIMIT = 1000;

    for (char c : text) {
        current_sentence += c;
        if (c == '.' || c == '?' || c == '!' || c == '\n') {
            if (!current_sentence.empty()) {
                sentences.push_back(current_sentence);
                current_sentence = "";
            }
        }
        else if (current_sentence.length() > SAFE_LIMIT && isspace(c)) {
            if (!current_sentence.empty()) {
                sentences.push_back(current_sentence);
                current_sentence = "";
            }
        }
        else if (current_sentence.length() >= MAX_LIMIT) {
            sentences.push_back(current_sentence);
            current_sentence = "";
        }
    }
    if (!current_sentence.empty()) {
        sentences.push_back(current_sentence);
    }
    return sentences;
}

// UTF8 -> Wstring 변환
wstring StringToWstring(const string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// 실제 번역을 수행하는 로직 (버튼 클릭 시 호출)
void PerformTranslation(int target_lang_id, int source_lang_id) {
    if (!g_translator || !g_processor) return;

    total_text = ""; // 결과 초기화

    if (target_lang_id == source_lang_id) {
        total_text = input_text;
    }
    else {
        vector<string> sentences = SplitSentences(input_text);

        string start_token = (source_lang_id == 1) ? "kor_Hang" : ((source_lang_id == 3) ? "zho_Hans" : "eng_Latn");
        string target_token_str = (target_lang_id == 1) ? "kor_Hang" : ((target_lang_id == 2) ? "eng_Latn" : "zho_Hans");

        size_t batch_size = 8;
        for (size_t i = 0; i < sentences.size(); i += batch_size) {
            vector<vector<string>> batch_input;
            for (size_t j = i; j < min(i + batch_size, sentences.size()); ++j) {
                vector<string> tokens;
                g_processor->Encode(sentences[j], &tokens);
                if (tokens.empty()) continue;
                tokens.insert(tokens.begin(), start_token);
                tokens.push_back("</s>");
                batch_input.push_back(tokens);
            }

            if (batch_input.empty()) continue;

            vector<vector<string>> target_prefix(batch_input.size(), { target_token_str });
            ctranslate2::TranslationOptions options;
            options.max_decoding_length = 1024;
            options.beam_size = 5;

            auto results = g_translator->translate_batch(batch_input, target_prefix, options);

            for (const auto& res : results) {
                vector<string> output_tokens = res.output();
                if (!output_tokens.empty()) output_tokens.erase(output_tokens.begin());
                string translated_sent;
                g_processor->Decode(output_tokens, &translated_sent);
                total_text += translated_sent + " ";
            }
        }
    }
    // 화면 업데이트
    wstring wResult = StringToWstring(total_text);
    SetWindowText(hResultStatic, wResult.c_str());
}

static HFONT hFont = NULL;

LRESULT CALLBACK MainDialogProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        hFont = CreateFont(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");

        // 버튼 생성
        CreateWindow(L"BUTTON", L"Korean", WS_VISIBLE | WS_CHILD, 10, 10, 90, 40, hwnd, (HMENU)1, NULL, NULL);
        CreateWindow(L"BUTTON", L"English", WS_VISIBLE | WS_CHILD, 110, 10, 90, 40, hwnd, (HMENU)2, NULL, NULL);
        CreateWindow(L"BUTTON", L"Chinese", WS_VISIBLE | WS_CHILD, 210, 10, 90, 40, hwnd, (HMENU)0, NULL, NULL);

        hResultStatic = CreateWindow(L"EDIT", L"Select Target Language",
            WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL | ES_LEFT,
            10, 70, 560, 480, hwnd, NULL, NULL, NULL);

        if (hResultStatic && hFont) {
            SendMessage(hResultStatic, WM_SETFONT, (WPARAM)hFont, TRUE);
        }
        break;

    case WM_COMMAND:
        if (HIWORD(wp) == BN_CLICKED) {
            int selected_id = LOWORD(wp);
            SetWindowText(hResultStatic, L"Translating...");
            PerformTranslation(selected_id, source_lang_id);
        }
        break;

    case WM_DESTROY:
        // 4. 생성한 폰트 오브젝트 메모리 해제
        if (hFont) {
            DeleteObject(hFont);
        }
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, msg, wp, lp);
    }
    return 0;
}

int main(int argc, char** argv) {
    _putenv_s("KMP_DUPLICATE_LIB_OK", "TRUE");
    system("chcp 65001 > nul");

    string line;
    string all_data;
    while (getline(cin, line)) {
        all_data += line + "\n";
    }
    if (all_data.empty()) return 0;

    stringstream ss(all_data);

    if (!(ss >> source_lang_id)) source_lang_id = 2;

    char c;
    if (ss.good()) {
        char c;
        while (ss.get(c) && isspace(c));
        if (ss) {
            input_text += c;
            string rest;
            getline(ss, rest, '\0');
            input_text += rest;
        }
    }


    // 2. 모델 로드 (전역 변수에 할당)
    const string model_path = "./nllb_ct2";
    const string sp_model_path = "./nllb_ct2/sentencepiece.bpe.model";

    ctranslate2::Translator translator(model_path, ctranslate2::Device::CPU);
    sentencepiece::SentencePieceProcessor processor;
    if (!processor.Load(sp_model_path).ok()) return 1;

    g_translator = &translator;
    g_processor = &processor;

    // 3. 윈도우 생성 및 실행
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = MainDialogProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"TransMainWin";
    RegisterClass(&wc);

    HWND hwnd = CreateWindow(L"TransMainWin", L"NLLB Translation System",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 600, 600, NULL, NULL, wc.hInstance, NULL);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}