#ifndef BMP_H
#define BMP_H

#include <string>
#include <cstdio>

class BMP
{
    public:
        static void saveImage(std::string filename, unsigned char* data, int w, int h, int c);
        BMP(std::string filename, size_t w, size_t h);
        BMP(std::string filename);
        void close();
        void writeRow();
        void readRow();
        inline void pixel(size_t x, int r, int g, int b) {
            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;
            row_data[x].r = r;
            row_data[x].g = g;
            row_data[x].b = b;
        }
        int width, height;
        #pragma pack(1)
        struct Pixel { unsigned char b,g,r; };
        #pragma pack()
        Pixel* row_data;
    private:
        FILE* bmp_file;
};

#endif // BMP_H
