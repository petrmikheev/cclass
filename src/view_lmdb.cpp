#include <cstdio>
#include "lmdb_reader.h"
#include "BMP.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Using: view_lmdb <LMDB_DIR> <id> <output>.bmp\n");
        return 1;
    }
    size_t id;
    sscanf(argv[2], "%zu", &id);
    LMDB db(argv[1]);
    printf("LMDB size: %zu\n", db.getSize());
    int w, h, c, l;
    unsigned char* data;
    for (size_t i = 0; i <= id; ++i) data = db.getImageData(&w, &h, &c, &l);
    BMP::saveImage(argv[3], data, w, h, c);
    printf("w = %d\nh = %d\nc = %d\nlabel = %d\n", w, h, c, l);
    return 0;
}

