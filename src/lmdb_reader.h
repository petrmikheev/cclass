#ifndef LMDB_READER_H
#define LMDB_READER_H

#include <string>
#include <lmdb.h>

class LMDB {
    public:
        LMDB(std::string filename);
        ~LMDB();
        unsigned char* getImageData(int* w, int* h, int* c, int* label);
        inline size_t getSize() { return stat.ms_entries; }
    private:
        MDB_env *env;
        MDB_dbi dbi;
        MDB_txn *txn;
        MDB_cursor *cursor;
        MDB_stat stat;
        size_t pos;
};

#endif // LMDB_READER_H
