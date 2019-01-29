#ifndef _METRIC_H
#define _METRIC_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

#include <stdint.h>

#include "data_types.h"

class comparePkt_tuple{
public:
  bool operator()(const struct pkt_tuple &lhs, const struct pkt_tuple &rhs){
    if(lhs.src_ip < rhs.src_ip) return true;
    if(lhs.src_ip > rhs.src_ip) return false;
    if(lhs.dst_ip < rhs.dst_ip) return true;
    if(lhs.dst_ip > rhs.dst_ip) return false;
    if(lhs.proto < rhs.proto) return true;
    if(lhs.proto > rhs.proto) return false;
    if(lhs.src_port < rhs.src_port) return true;
    if(lhs.src_port > rhs.src_port) return false;
    if(lhs.dst_port < rhs.dst_port) return true;
    if(lhs.dst_port > rhs.dst_port) return false;
    return false;
  }
};

class Metric{
public:
  Metric(std::string file_name, int timeout, int idle): output(file_name), timeout(timeout*1000), idle_threshold(idle*1000) { }
  void process_pkt(struct pkt_tuple&, struct pkt_info&);
  void output_flow_features();
  void clean_flow_stats(uint64_t);
private:
  std::ofstream output;
  int timeout; //ms
  int idle_threshold;
  std::map<struct pkt_tuple, struct features, comparePkt_tuple> flow_stats;
  std::mutex flow_stats_mutex;
  void add_new_entry(struct pkt_tuple&, struct pkt_info&);
  void update_entry(struct features&, bool, struct pkt_info&);
  //void output_flow(const struct pkt_tuple&, struct features&);
};

#endif
