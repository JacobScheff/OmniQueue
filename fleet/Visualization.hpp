#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <string>

#include "City.hpp"

class Visualization {
public:
    explicit Visualization(const City& city, int windowWidth = 800, int windowHeight = 800)
        : city_(city), windowWidth_(windowWidth), windowHeight_(windowHeight) {}

    // Opens a window, draws intersections and streets, and blocks until closed.
    void show() const {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &Visualization::WndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = kClassName;

        if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            return;
        }

        const DWORD style = WS_OVERLAPPEDWINDOW;
        RECT rect{0, 0, windowWidth_, windowHeight_};
        AdjustWindowRect(&rect, style, FALSE);

        HWND hwnd = CreateWindowExW(
            0,
            kClassName,
            L"Fleet City Visualization",
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            nullptr,
            nullptr,
            GetModuleHandleW(nullptr),
            const_cast<Visualization*>(this));

        if (!hwnd) {
            return;
        }

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        MSG msg;
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

private:
    static constexpr const wchar_t* kClassName = L"FleetCityVisualization";
    static constexpr int kMargin = 24;
    static constexpr int kIntersectionRadius = 4;

    const City& city_;
    int windowWidth_;
    int windowHeight_;

    void paint(HWND hwnd) const {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client{};
        GetClientRect(hwnd, &client);
        const int drawW = std::max(1, static_cast<int>(client.right - client.left) - 2 * kMargin);
        const int drawH = std::max(1, static_cast<int>(client.bottom - client.top) - 2 * kMargin);

        HBRUSH bg = CreateSolidBrush(RGB(245, 245, 248));
        FillRect(hdc, &client, bg);
        DeleteObject(bg);

        HPEN streetPen = CreatePen(PS_SOLID, 2, RGB(90, 110, 140));
        HGDIOBJ oldPen = SelectObject(hdc, streetPen);

        for (const Street& street : city_.streets) {
            if (!street.I1 || !street.I2) continue;
            int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            mapPoint(street.I1->pos.x, street.I1->pos.y, drawW, drawH, x1, y1);
            mapPoint(street.I2->pos.x, street.I2->pos.y, drawW, drawH, x2, y2);
            MoveToEx(hdc, x1, y1, nullptr);
            LineTo(hdc, x2, y2);
        }

        SelectObject(hdc, oldPen);
        DeleteObject(streetPen);

        HBRUSH nodeBrush = CreateSolidBrush(RGB(220, 70, 70));
        HPEN nodePen = CreatePen(PS_SOLID, 1, RGB(120, 30, 30));
        oldPen = SelectObject(hdc, nodePen);
        HGDIOBJ oldBrush = SelectObject(hdc, nodeBrush);

        for (const Intersection* intersection : city_.intersections) {
            if (!intersection) continue;
            int x = 0, y = 0;
            mapPoint(intersection->pos.x, intersection->pos.y, drawW, drawH, x, y);
            Ellipse(
                hdc,
                x - kIntersectionRadius,
                y - kIntersectionRadius,
                x + kIntersectionRadius,
                y + kIntersectionRadius);
        }

        SelectObject(hdc, oldBrush);
        SelectObject(hdc, oldPen);
        DeleteObject(nodeBrush);
        DeleteObject(nodePen);

        std::wstring summary =
            L"Intersections: " + std::to_wstring(city_.intersections.size()) +
            L"   Streets: " + std::to_wstring(city_.streets.size());
        SetBkMode(hdc, TRANSPARENT);
        TextOutW(hdc, kMargin, 4, summary.c_str(), static_cast<int>(summary.size()));

        EndPaint(hwnd, &ps);
    }

    void mapPoint(int cityX, int cityY, int drawW, int drawH, int& outX, int& outY) const {
        const double sx = city_.width > 1 ? static_cast<double>(cityX) / (city_.width - 1) : 0.5;
        const double sy = city_.height > 1 ? static_cast<double>(cityY) / (city_.height - 1) : 0.5;
        outX = kMargin + static_cast<int>(sx * drawW);
        outY = kMargin + static_cast<int>(sy * drawH);
    }

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        Visualization* self = nullptr;
        if (msg == WM_NCCREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
            self = static_cast<Visualization*>(cs->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        } else {
            self = reinterpret_cast<Visualization*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }

        switch (msg) {
            case WM_PAINT:
                if (self) self->paint(hwnd);
                return 0;
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
            default:
                return DefWindowProcW(hwnd, msg, wParam, lParam);
        }
    }
};

#endif
