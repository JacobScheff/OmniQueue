#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "City.hpp"

class Visualization {
public:
    explicit Visualization(const City& city, int imageWidth = 800, int imageHeight = 800)
        : city_(city), imageWidth_(imageWidth), imageHeight_(imageHeight) {}

    // Renders intersections and streets to a BMP and opens it in the default viewer.
    void show(const std::string& path = "city.bmp") const {
        writeBmp(path);
        std::string command = "cmd /c start \"\" \"" + path + "\"";
        std::system(command.c_str());
    }

private:
    static constexpr int kMargin = 24;
    static constexpr int kIntersectionRadius = 4;

    const City& city_;
    int imageWidth_;
    int imageHeight_;

    void mapPoint(int cityX, int cityY, int& outX, int& outY) const {
        const int drawW = std::max(1, imageWidth_ - 2 * kMargin);
        const int drawH = std::max(1, imageHeight_ - 2 * kMargin);
        const double sx = city_.width > 1 ? static_cast<double>(cityX) / (city_.width - 1) : 0.5;
        const double sy = city_.height > 1 ? static_cast<double>(cityY) / (city_.height - 1) : 0.5;
        outX = kMargin + static_cast<int>(sx * drawW);
        outY = kMargin + static_cast<int>(sy * drawH);
    }

    static void putPixel(std::vector<uint8_t>& pixels, int w, int h, int x, int y,
                         uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        const size_t i = (static_cast<size_t>(y) * w + x) * 3;
        pixels[i] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }

    static void drawLine(std::vector<uint8_t>& pixels, int w, int h,
                         int x0, int y0, int x1, int y1,
                         uint8_t r, uint8_t g, uint8_t b) {
        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true) {
            putPixel(pixels, w, h, x0, y0, r, g, b);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    static void fillCircle(std::vector<uint8_t>& pixels, int w, int h,
                           int cx, int cy, int radius,
                           uint8_t r, uint8_t g, uint8_t b) {
        const int r2 = radius * radius;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (x * x + y * y <= r2) {
                    putPixel(pixels, w, h, cx + x, cy + y, r, g, b);
                }
            }
        }
    }

    void writeBmp(const std::string& path) const {
        const int w = imageWidth_;
        const int h = imageHeight_;
        std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3, 245);

        for (const Street& street : city_.streets) {
            if (!street.I1 || !street.I2) continue;
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            mapPoint(street.I1->pos.x, street.I1->pos.y, x0, y0);
            mapPoint(street.I2->pos.x, street.I2->pos.y, x1, y1);
            drawLine(pixels, w, h, x0, y0, x1, y1, 90, 110, 140);
        }

        for (const Intersection* intersection : city_.intersections) {
            if (!intersection) continue;
            int x = 0, y = 0;
            mapPoint(intersection->pos.x, intersection->pos.y, x, y);
            fillCircle(pixels, w, h, x, y, kIntersectionRadius, 220, 70, 70);
        }

        const int rowStride = ((w * 3 + 3) / 4) * 4;
        const uint32_t pixelBytes = static_cast<uint32_t>(rowStride) * h;
        const uint32_t fileSize = 54 + pixelBytes;

        std::ofstream out(path, std::ios::binary);
        auto write16 = [&](uint16_t v) {
            out.put(static_cast<char>(v & 0xFF));
            out.put(static_cast<char>((v >> 8) & 0xFF));
        };
        auto write32 = [&](uint32_t v) {
            out.put(static_cast<char>(v & 0xFF));
            out.put(static_cast<char>((v >> 8) & 0xFF));
            out.put(static_cast<char>((v >> 16) & 0xFF));
            out.put(static_cast<char>((v >> 24) & 0xFF));
        };

        out.put('B');
        out.put('M');
        write32(fileSize);
        write32(0);
        write32(54);
        write32(40);
        write32(static_cast<uint32_t>(w));
        write32(static_cast<uint32_t>(h));
        write16(1);
        write16(24);
        write32(0);
        write32(pixelBytes);
        write32(2835);
        write32(2835);
        write32(0);
        write32(0);

        std::vector<char> row(static_cast<size_t>(rowStride), 0);
        for (int y = h - 1; y >= 0; --y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3;
                row[static_cast<size_t>(x) * 3] = static_cast<char>(pixels[i + 2]);
                row[static_cast<size_t>(x) * 3 + 1] = static_cast<char>(pixels[i + 1]);
                row[static_cast<size_t>(x) * 3 + 2] = static_cast<char>(pixels[i]);
            }
            out.write(row.data(), rowStride);
        }
    }
};

#endif
