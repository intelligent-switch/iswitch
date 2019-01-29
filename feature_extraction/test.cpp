#include <iostream>
#include <fstream>
#include <cstdint>

#include "data_types.h"

extern "C" void test_pass_pkt(struct pkt_tuple, struct pkt_info, unsigned);

void test_pass_pkt(struct pkt_tuple tuple, struct pkt_info info, unsigned lcore_id){
  static std::ofstream output("received_info");

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

  output << lcore_id << ' ';
  output << a1 << '.' << b1 << '.' << c1 << '.' << d1 << "  " << a2 << '.' << b2 << '.' << c2 << '.' << d2;
  output << ' ' << (int)tuple.proto << ' ' << tuple.src_port << ' ' << tuple.dst_port << ' ' << info.data_len << ' ' << info.timestamp << '\n';
}
