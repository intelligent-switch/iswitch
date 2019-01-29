#ifndef _DATA_TYPES_H
#define _DATA_TYPES_H

#include <stdint.h>

#define UDP_HEADER_LEN 42
#define TCP_HEADER_LEN 54

#define numBlocks 32 
#define threadsPerBlock 32
#define pktsPerThread 64

#define TotalThreads (numBlocks * threadsPerBlock)
#define ArraySize (TotalThreads * pktsPerThread)

#define IDLE_THRESHOLD 2000

#define TIMEOUT_THRESHOLD 5000

#define HASH_ENTRIES 5000000

#define OUTPUT_INTERVAL 5
#define CLEAN_INTERVAL (TIMEOUT_THRESHOLD / 1000 * 2)

struct pkt_tuple{
  uint32_t src_ip;
  uint32_t dst_ip;
  uint8_t proto;
  uint16_t src_port;
  uint16_t dst_port;
};

struct pkt_info{
  uint16_t data_len;
  uint64_t timestamp;
  bool psh;
  bool urg;
};

struct pkt_tuple_info{
  uint32_t hash;
  struct pkt_tuple tuple;
  struct pkt_info info;

  int next_index;
};

struct flow_features{
  uint64_t timestamp;

  long total_packets; //Total packets
  long total_volume; //Total bytes
  int min_pktl; //The size of the smallest packet sent (in bytes)
  int max_pktl; //The size of the largest packet sent (in bytes)
  int sqsum_pktl;
  int min_iat; //The minimum amount of time between two packets sent (in microseconds)
  int max_iat;
  int sum_iat;
  int sqsum_iat;

  int psh_cnt;
  int urg_cnt;
  int total_hlen;
};

struct features{
  struct flow_features forward;
  struct flow_features backward;

  int hlen;

  uint64_t first_timestamp;
  uint64_t last_timestamp;
  uint64_t active_timestamp;
  int min_active;
  int max_active;
  int sum_active;
  int sqsum_active;
  int active_times;
  int min_idle;
  int max_idle;
  int sum_idle;
  int sqsum_idle;
  int idle_times;
};

struct hash_entry{
  uint32_t hash;
  struct pkt_tuple tuple;
  struct features bidirec_flow;
  int state;
  int mutex;
};

#endif
