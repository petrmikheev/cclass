#include <cstdio>
#include <cstdlib>
#include "lmdb_reader.h"

LMDB::LMDB(std::string filename) {
    mdb_env_create(&env);
    mdb_env_open(env, filename.c_str(), 0, 0664);
    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_open(txn, NULL, 0, &dbi);
    mdb_stat (txn, dbi, &stat);
    mdb_cursor_open(txn, dbi, &cursor);
    pos = 0;
}

LMDB::~LMDB() {
    mdb_cursor_close(cursor);
    mdb_close(env, dbi);
    mdb_env_close(env);
}

static size_t unpackVarint(char*& buf) {
    size_t ans = 0;
    size_t base = 1;
    while (*buf < 0) { ans += base * (*(buf++))&127; base *= 128; }
    return ans + base * *(buf++);
}

unsigned char* LMDB::getImageData(int* w, int* h, int* c, int* label) {
    MDB_val key, data;
    unsigned char* ans = NULL;
    pos = (pos+1) % getSize();
    mdb_cursor_get(cursor, &key, &data, pos ? MDB_NEXT : MDB_FIRST);
    char* buf = (char*)data.mv_data;
    char* buf_end = buf + data.mv_size;
    while (buf < buf_end) {
        int type = *buf & 7;
        int fnum = (unsigned char)*(buf++) >> 3;
        if (type == 0) {
            size_t v = unpackVarint(buf);
            if (fnum == 1) *c = v;
            else if (fnum == 2) *h = v;
            else if (fnum == 3) *w = v;
            else if (fnum == 5) *label = v;
        } else if (type == 1) buf += 8;
        else if (type == 5) buf += 4;
        else if (type == 2) {
            size_t s = unpackVarint(buf);
            ans = (unsigned char*)buf;
            buf += s;
        } else { printf("Unknown type\n"); exit(1); }
    }
    if (buf != buf_end || ans == NULL) { printf("Unexpected end\n"); exit(1); }
    return ans;
}

