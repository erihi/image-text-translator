#pragma once

#include <windows.h>
#include <string>
#include <vector>

extern std::string g_droppedFilePath;
extern std::string g_lang;

void setup_console_output();

// 유틸리티: Wide String(UTF-16) -> UTF-8 String 변환 함수
std::string ToUtf8(const std::wstring& wstr);

class DragDropWindow
{
private:
	HWND m_hWnd = NULL;
	HINSTANCE m_hInstance;
	bool m_showDropArea = false;
	static constexpr const wchar_t* CLASS_NAME = L"DragDropWindow";

public:
	DragDropWindow(HINSTANCE hInstance);
	bool Initialize(const wchar_t* title, int width, int height);
	int RunMessageLoop();

private:
	LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT OnDropFiles(WPARAM wParam);
	static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	bool RegisterClass();
};