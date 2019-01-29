#include "data_types.h"
#include <iostream>

extern std::ostream &operator<<(std::ostream &os, const pkt_tuple &tuple){
  uint32_t a1, b1, c1, d1;
	uint32_t a2, b2, c2, d2;

	a1 = (tuple.src_ip >> 24) & 0xff;
	b1 = (tuple.src_ip >> 16) & 0xff;
	c1 = (tuple.src_ip >> 8) & 0xff;
	d1 = (tuple.src_ip) & 0xff;
	a2 = (tuple.dst_ip >> 24) & 0xff;
	b2 = (tuple.dst_ip >> 16) & 0xff;
	c2 = (tuple.dst_ip >> 8) & 0xff;
	d2 = (tuple.dst_ip) & 0xff;

  os << a1 << '.' << b1 << '.' << c1 << '.' << d1 << ' ' << a2 << '.' << b2 << '.' << c2 << '.' << d2;
  os << ' ' << (int)tuple.proto << ' ' << tuple.src_port << ' ' << tuple.dst_port;
  return os;
}

extern std::ostream &operator<<(std::ostream &os, const flow_features &flow){
  int mean_pktl = 0;
  int std_pktl = 0;
  if(flow.total_packets > 0){
    mean_pktl = flow.total_volume / flow.total_packets;
    std_pktl = (flow.sqsum_pktl - 2*flow.total_volume*mean_pktl) / flow.total_packets + mean_pktl * mean_pktl;
  }

  int mean_iat = 0;
  int std_iat = 0;
  if(flow.total_packets > 1){
    mean_iat = flow.sum_iat / (flow.total_packets - 1);
    std_iat = (flow.sqsum_iat - 2*flow.sum_iat*mean_iat) / (flow.total_packets - 1) + mean_iat * mean_iat;
  }

  os << flow.total_packets << ' ' << flow.total_volume << ' ' << flow.min_pktl << ' '
    << mean_pktl << ' ' << flow.max_pktl << ' ' << std_pktl << ' ' <<
    flow.min_iat << ' ' << mean_iat << ' ' << flow.max_iat << ' ' << std_iat
    << ' ' << flow.psh_cnt << ' ' << flow.urg_cnt << ' ' << flow.total_hlen;
  return os;
}

extern std::ostream &operator<<(std::ostream &os, const features &bidirec_flow){
  int sflow_fpackets = 0;
  int sflow_fbytes = 0;
  int sflow_bpackets = 0;
  int sflow_bbytes = 0;

  int mean_active = 0;
  int std_active = 0;
  if(bidirec_flow.active_times > 0){
    mean_active = bidirec_flow.sum_active / bidirec_flow.active_times;
    std_active = (bidirec_flow.sqsum_active - 2*bidirec_flow.sum_active*mean_active) / bidirec_flow.active_times + mean_active * mean_active;

    sflow_fpackets = bidirec_flow.forward.total_packets / bidirec_flow.active_times;
    sflow_fbytes = bidirec_flow.forward.total_volume / bidirec_flow.active_times;
    sflow_bpackets = bidirec_flow.backward.total_packets / bidirec_flow.active_times;
    sflow_bbytes = bidirec_flow.backward.total_volume / bidirec_flow.active_times;
  }

  int mean_idle = 0;
  int std_idle = 0;
  if(bidirec_flow.idle_times > 0){
    mean_idle = bidirec_flow.sum_idle / bidirec_flow.idle_times;
    std_idle = (bidirec_flow.sqsum_idle - 2*bidirec_flow.sum_idle*mean_idle) / bidirec_flow.idle_times + mean_idle * mean_idle;
  }

  unsigned long duration = bidirec_flow.last_timestamp - bidirec_flow.first_timestamp;

  os << bidirec_flow.forward << " " << bidirec_flow.backward << " " << duration << " " << bidirec_flow.min_active <<
        " " << mean_active << " " << bidirec_flow.max_active << " " << std_active << " " <<
        bidirec_flow.min_idle << " " << mean_idle << " " << bidirec_flow.max_idle << " " << std_idle
        << " " << sflow_fpackets << " " << sflow_fbytes << " " << sflow_bpackets << " " << sflow_bbytes;
  return os;
}
