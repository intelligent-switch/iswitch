#ifndef _GPU_METRIC_H
#define _GPU_METRIC_H

#include "data_types.h"

#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <queue>
#include <mutex>

#define PORT 40000

class GPU_Metric {
public:
  GPU_Metric();
  ~GPU_Metric();

  void add_pkts(struct pkt_tuple_info *cpu_pkts){
    q_m.lock();
    q.push(cpu_pkts);
    ++array_num;
    if(!has_data) has_data = true;
    q_m.unlock();
  }

  void process();

  void gpu_output_stats();
  void output_to_file();
  void output_through_udp();

  void quit(){
    force_quit = true;
  }

  void clean_flow_stats(uint64_t timestamp);
private:
  volatile bool force_quit;

  std::queue<struct pkt_tuple_info *> q;
  std::mutex q_m;
  int array_num;
  int max_array_num;
  volatile bool has_data;

  int *cpu_first_index;
  int *cpu_last_index;
  int *gpu_first_index;
  struct pkt_tuple_info *gpu_pkts;

  hash_entry *hash_table;
  hash_entry *gpu_hash_table[2]; // update hash table
  hash_entry *cpu_hash_table;
  volatile bool copy_to_cpu;

  volatile uint64_t update_timestamp;
  volatile bool update_hash_table;

  int sockfd;
  struct sockaddr_in servaddr;

  void process_pkts(struct pkt_tuple_info *);
};

#endif
