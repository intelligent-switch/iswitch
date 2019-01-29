#include "Metric.h"

#include <cmath>

std::ostream &operator<<(std::ostream &os, const pkt_tuple &tuple);

std::ostream &operator<<(std::ostream &os, const flow_features &flow);

std::ostream &operator<<(std::ostream &os, const features &bidirec_flow);

using namespace std;

struct pkt_tuple inverse_pkt_tuple(struct pkt_tuple &tuple){
  struct pkt_tuple ret;
  ret.src_ip = tuple.dst_ip;
  ret.dst_ip = tuple.src_ip;
  ret.proto = tuple.proto;
  ret.src_port = tuple.dst_port;
  ret.dst_port = tuple.src_port;

  return ret;
}

void Metric::process_pkt(struct pkt_tuple &tuple, struct pkt_info &info){
  auto it1 = flow_stats.find(tuple);
  auto it2 = flow_stats.find(inverse_pkt_tuple(tuple));
  if(it1 == flow_stats.end() && it2 == flow_stats.end()){
    flow_stats_mutex.lock();
    add_new_entry(tuple, info);
    flow_stats_mutex.unlock();
  }
  else if(it1 != flow_stats.end()){
    flow_stats_mutex.lock();
    update_entry(it1->second, true, info);
    flow_stats_mutex.unlock();
  }
  else{
    flow_stats_mutex.lock();
    update_entry(it2->second, false, info);
    flow_stats_mutex.unlock();
  }
}

void Metric::add_new_entry(struct pkt_tuple &tuple, struct pkt_info &info){
  struct features &flow = flow_stats[tuple];

  flow = {0};

  switch(tuple.proto){
    case 6: flow.hlen = UDP_HEADER_LEN; break;
    case 17: flow.hlen = TCP_HEADER_LEN; break;
  }

  flow.first_timestamp = info.timestamp;
  flow.last_timestamp = info.timestamp;
  flow.active_timestamp = info.timestamp;

  flow.forward.timestamp = info.timestamp;
  flow.forward.total_packets = 1;
  flow.forward.total_volume = info.data_len;
  flow.forward.min_pktl = info.data_len;
  flow.forward.max_pktl = info.data_len;
  flow.forward.sqsum_pktl = info.data_len * info.data_len;

  if(info.psh) ++flow.forward.psh_cnt;
  if(info.urg) ++flow.forward.urg_cnt;
  flow.forward.total_hlen += flow.hlen;
}

void Metric::update_entry(struct features &bidirec_flow, bool is_forward, struct pkt_info &info){
  int diff = info.timestamp - bidirec_flow.last_timestamp;
  if(diff > idle_threshold){
    int cur_active = bidirec_flow.last_timestamp - bidirec_flow.active_timestamp;
    if(cur_active > 0){
      if(cur_active < bidirec_flow.min_active || bidirec_flow.min_active == 0) bidirec_flow.min_active = cur_active;
      if(cur_active > bidirec_flow.max_active) bidirec_flow.max_active = cur_active;
      bidirec_flow.sum_active += cur_active;
      bidirec_flow.sqsum_active += cur_active * cur_active;
      ++bidirec_flow.active_times;
    }
    if(diff < bidirec_flow.min_idle || bidirec_flow.min_idle == 0) bidirec_flow.min_idle = diff;
    if(diff > bidirec_flow.max_idle) bidirec_flow.max_idle = diff;
    bidirec_flow.sum_idle += diff;
    bidirec_flow.sqsum_idle += diff * diff;
    ++bidirec_flow.idle_times;
    bidirec_flow.active_timestamp = info.timestamp;
  }
  bidirec_flow.last_timestamp = info.timestamp;

  struct flow_features *flow;
  if(is_forward) flow = &(bidirec_flow.forward);
  else flow = &(bidirec_flow.backward);

  flow->total_hlen += bidirec_flow.hlen;
  if(info.psh) ++flow->psh_cnt;
  if(info.urg) ++flow->urg_cnt;

  if(flow->total_packets == 0){
    flow->timestamp = info.timestamp;
    flow->total_packets = 1;
    flow->total_volume = info.data_len;
    flow->min_pktl = info.data_len;
    flow->max_pktl = info.data_len;
    flow->sqsum_pktl = info.data_len * info.data_len;
    return;
  }

  ++flow->total_packets;
  flow->total_volume += info.data_len;
  if(flow->min_pktl > info.data_len) flow->min_pktl = info.data_len;
  if(flow->max_pktl < info.data_len) flow->max_pktl = info.data_len;
  flow->sqsum_pktl += info.data_len * info.data_len;

  int interval = info.timestamp - flow->timestamp;
  flow->timestamp = info.timestamp;
  if(flow->total_packets == 2){
    flow->min_iat = interval;
    flow->max_iat = interval;
    flow->sum_iat = interval;
    flow->sqsum_iat = interval * interval;
  }
  else{
    if(flow->min_iat > interval) flow->min_iat = interval;
    if(flow->max_iat < interval) flow->max_iat = interval;
    flow->sum_iat += interval;
    flow->sqsum_iat += interval * interval;
  }
}

void Metric::output_flow_features(){
  map<struct pkt_tuple, struct features, comparePkt_tuple> cur_stats(flow_stats);

  for(auto it = cur_stats.begin(); it != cur_stats.end(); ++it){
    //output_flow(it->first, it->second);
    output << it->first << " " << it->second << endl;
  }
}

// ostream &operator<<(ostream &out, const struct flow_features &flow){
//   out << flow.total_packets << ' ' << flow.total_volume << ' ' << flow.min_pktl << ' '
//     << flow.mean_pktl << ' ' << flow.max_pktl << ' ' << flow.std_pktl << ' ' <<
//     flow.min_iat << ' ' << flow.mean_iat << ' ' << flow.max_iat << ' ' << flow.std_iat
//     << ' ' << flow.psh_cnt << ' ' << flow.urg_cnt << ' ' << flow.total_hlen;
//
//   return out;
// }
//
// void Metric::output_flow(const struct pkt_tuple &tuple, struct features &flow){
//   uint32_t a1, b1, c1, d1;
// 	uint32_t a2, b2, c2, d2;
//
// 	a1 = (tuple.src_ip >> 24) & 0xff;
// 	b1 = (tuple.src_ip >> 16) & 0xff;
// 	c1 = (tuple.src_ip >> 8) & 0xff;
// 	d1 = (tuple.src_ip) & 0xff;
// 	a2 = (tuple.dst_ip >> 24) & 0xff;
// 	b2 = (tuple.dst_ip >> 16) & 0xff;
// 	c2 = (tuple.dst_ip >> 8) & 0xff;
// 	d2 = (tuple.dst_ip) & 0xff;
//
//   output << a1 << '.' << b1 << '.' << c1 << '.' << d1 << ' ' << a2 << '.' << b2 << '.' << c2 << '.' << d2;
//   output << ' ' << (int)tuple.proto << ' ' << tuple.src_port << ' ' << tuple.dst_port << ' ';
//
//   output << flow.forward << ' ' << flow.backward << endl;
// }

void Metric::clean_flow_stats(uint64_t timestamp){
  map<struct pkt_tuple, struct features, comparePkt_tuple> cur_stats(flow_stats);

  uint64_t clean_time = timestamp - timeout;

  for(auto it = cur_stats.begin(); it != cur_stats.end(); ++it){
    if(it->second.forward.timestamp < clean_time && it->second.backward.timestamp < clean_time){
      flow_stats_mutex.lock();
      flow_stats.erase(it->first);
      flow_stats_mutex.unlock();
    }
  }
}
