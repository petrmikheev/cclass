#include "BMP.h"
#include <stdint.h>
#include <cstdlib>
#include <assert.h>

void BMP::saveImage(std::string filename, unsigned char* data, int w, int h, int c) {
    BMP bmp(filename, w, h);
    for (int y=0; y<h; ++y) {
        for (int x=0; x<w; ++x) {
            unsigned char* cell = data + (w*(h-y-1) + x) * c;
            if (c==3) bmp.pixel(x, cell[0], cell[1], cell[2]);
            else bmp.pixel(x, cell[0], cell[0], cell[0]);
        }
        bmp.writeRow();
    }
    bmp.close();
}

BMP::BMP(std::string filename) {
    bmp_file = fopen(filename.c_str(), "rb");
    int dataPos;
    short bp;
    fseek(bmp_file, 0xA, SEEK_SET);
    int count = fread(&dataPos, 4, 1, bmp_file);
    fseek(bmp_file, 0x12, SEEK_SET);
    count += fread(&width, 4, 1, bmp_file);
    count += fread(&height, 4, 1, bmp_file);
    fseek(bmp_file, 0x1c, SEEK_SET);
    count += fread(&bp, 2, 1, bmp_file);
    if (count != 4) {
        printf("BMP: Unexpected end of file\n");
        exit(1);
    }
    if (bp!=24) {
        printf("BMP: Input file format doesn`t supported");
        exit(1);
    }
    row_data = new Pixel[width+1];
    fseek(bmp_file, dataPos, SEEK_SET);
}

BMP::BMP(std::string filename, size_t w, size_t h) {
    if (w % 4) w += 4 - (w % 4);
    width = w;
    bmp_file = fopen(filename.c_str(), "wb");
    #pragma pack(1)
    struct {
        uint16_t type; // BM
        uint32_t file_size; // data_size + 54
        uint32_t reserved; // 0
        uint32_t data_offset; // 54
        uint32_t biSize; // 40
        uint32_t biWidth;
        uint32_t biHeight;
        uint16_t biPlanes; // 1
        uint16_t bitsPerPixel; // 24
        uint32_t biCompression; // 0
        uint32_t data_size;
        uint64_t zero1;
        uint64_t zero2;
    } bmp_header;

    #pragma pack()
    size_t data_size = w * h * 3;
    bmp_header.type = 0x4d42;
    bmp_header.file_size = data_size + 54;
    bmp_header.reserved = bmp_header.zero1 = bmp_header.zero2 = 0;
    bmp_header.data_offset = 54;
    bmp_header.biSize = 40;
    bmp_header.biWidth = w;
    bmp_header.biHeight = h;
    bmp_header.biPlanes = 1;
    bmp_header.bitsPerPixel = 24;
    bmp_header.biCompression = 0;
    bmp_header.data_size = data_size;
    fwrite(&bmp_header, sizeof(bmp_header), 1, bmp_file);
    row_data = new Pixel[w];
}

void BMP::writeRow() {
    fwrite(row_data, 3, width, bmp_file);
}

void BMP::readRow() {
    assert(fread(row_data, 1, (width*3+3)&~3, bmp_file) == ((width*3+3)&~3));
}

void BMP::close() {
    fclose(bmp_file);
    delete [] row_data;
}

