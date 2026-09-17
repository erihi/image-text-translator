#include <windows.h>
#include <shellapi.h> // Drag & Drop API (DragQueryFile 등)
#include <iostream>
#include <string>
#include <vector>     // 버퍼 관리를 위해 추가

// 전역 변수: 드롭된 경로를 저장할 변수 (UTF-8 std::string으로 변경)
std::string g_droppedFilePath;
std::string g_lang;

// 콘솔을 UTF-8로 설정하는 함수
void setup_console_output();

// 유틸리티: Wide String(UTF-16) -> UTF-8 String 변환 함수
std::string ToUtf8(const std::wstring& wstr);

/**
 * @brief 드래그 앤 드롭 기능을 포함하는 Win32 창을 관리하는 클래스
 */
class DragDropWindow
{
private:
    HWND m_hWnd = NULL;
    HINSTANCE m_hInstance;
    bool m_showDropArea = false;
    static constexpr const wchar_t* CLASS_NAME = L"DragDropWindow";

public:
    // 생성자
    DragDropWindow(HINSTANCE hInstance);

    // 창 클래스를 등록하고, 창을 생성하는 함수
    bool Initialize(const wchar_t* title, int width, int height);

    // 메시지 루프를 실행하는 함수
    int RunMessageLoop();

private:
    // Win32 창 클래스 등록
    bool RegisterClass();

    // 정적 윈도우 프로시저 (Win32 콜백 함수)
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // 인스턴스 메시지 핸들러
    LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam);

    // WM_DROPFILES 메시지 처리 함수
    LRESULT OnDropFiles(WPARAM wParam);
};

// =============================================================
// 구현부 (Implementation)
// =============================================================

// 콘솔 출력 설정 구현 (UTF-8)
void setup_console_output() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}

// UTF-16(wstring)을 UTF-8(string)로 변환하는 헬퍼 함수
std::string ToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();

    // 변환에 필요한 버퍼 크기 계산
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);

    // 버퍼 할당 및 변환 수행
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);

    return strTo;
}

// -------------------------------------------------------------
// DragDropWindow 클래스 구현
// -------------------------------------------------------------

DragDropWindow::DragDropWindow(HINSTANCE hInstance)
    : m_hInstance(hInstance)
{
}

bool DragDropWindow::RegisterClass() 
{
    WNDCLASSEXW wc = { sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, StaticWndProc, 0, 0, m_hInstance,
                       LoadIcon(NULL, IDI_APPLICATION), LoadCursor(NULL, IDC_ARROW),
                       (HBRUSH)(COLOR_WINDOW + 1), NULL, CLASS_NAME, NULL };
    return RegisterClassExW(&wc) != 0;
}

bool DragDropWindow::Initialize(const wchar_t* title, int width, int height) {
    if (!RegisterClass()) return false;
    m_hWnd = CreateWindowExW(WS_EX_CLIENTEDGE, CLASS_NAME, title, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height, NULL, NULL, m_hInstance, this);
    if (!m_hWnd) return false;
    ShowWindow(m_hWnd, SW_SHOW);
    return true;
}

int DragDropWindow::RunMessageLoop() {
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}

LRESULT CALLBACK DragDropWindow::StaticWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    DragDropWindow* pThis = nullptr;

    if (msg == WM_NCCREATE) {
        CREATESTRUCT* pCreate = (CREATESTRUCT*)lParam;
        pThis = (DragDropWindow*)pCreate->lpCreateParams;
        SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)pThis);
        pThis->m_hWnd = hWnd;
    }
    else {
        pThis = (DragDropWindow*)GetWindowLongPtr(hWnd, GWLP_USERDATA);
    }

    if (pThis) {
        return pThis->HandleMessage(msg, wParam, lParam);
    }
    else {
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

LRESULT DragDropWindow::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE:
        CreateWindowW(L"BUTTON", L"Korean", WS_VISIBLE | WS_CHILD, 10, 10, 90, 40, m_hWnd, (HMENU)1, m_hInstance, NULL);
        CreateWindowW(L"BUTTON", L"English", WS_VISIBLE | WS_CHILD, 110, 10, 90, 40, m_hWnd, (HMENU)2, m_hInstance, NULL);
        CreateWindowW(L"BUTTON", L"Chinese", WS_VISIBLE | WS_CHILD, 210, 10, 90, 40, m_hWnd, (HMENU)3, m_hInstance, NULL);
        return 0;
    case WM_COMMAND: {
        int wmId = LOWORD(wParam);
        if (wmId >= 1 && wmId <= 3) {
            g_lang = std::to_string(wmId);
            m_showDropArea = true;
            DragAcceptFiles(m_hWnd, TRUE);
            InvalidateRect(m_hWnd, NULL, TRUE);
        }
        return 0;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(m_hWnd, &ps);
        if (m_showDropArea) {
            RECT rect = { 10, 60, 530, 530 };
            Rectangle(hdc, rect.left, rect.top, rect.right, rect.bottom);
            SetBkMode(hdc, TRANSPARENT);
            DrawTextW(hdc, L"Drop Image Here", -1, &rect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }
        EndPaint(m_hWnd, &ps);
        return 0;
    }
    case WM_DROPFILES: return OnDropFiles(wParam);
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProc(m_hWnd, msg, wParam, lParam);
}

LRESULT DragDropWindow::OnDropFiles(WPARAM wParam) {
    HDROP hDrop = (HDROP)wParam;

    UINT size = DragQueryFileW(hDrop, 0, NULL, 0);
    if (size > 0) {
        std::wstring wpath(size + 1, L'\0');
        DragQueryFileW(hDrop, 0, &wpath[0], size + 1);
        g_droppedFilePath = ToUtf8(wpath);
    }
    DragFinish(hDrop);
    DestroyWindow(m_hWnd);
    return 0;
}