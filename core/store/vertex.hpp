#pragma once

typedef unsigned int vertex_id_t;

typedef unsigned char server_id_t; // 1 byte
typedef unsigned int local_id_t;   // 4 bytes
typedef std::pair<server_id_t, local_id_t> ikey_t;
typedef std::pair<server_id_t, local_id_t> item_t;