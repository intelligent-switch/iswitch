#include "data_types.h"
#include "Metric.h"
#include "GPU_Metric.h"

#include <iostream>
#include <thread>

using namespace std;

extern "C" void pass_cpu_pkts(struct pkt_tuple_info *);

extern "C" void output_flow_features();

extern "C" void clean_flow_stats(uint64_t);

extern "C" void quit_gpu_processing();

extern "C" void gpu_process();

//cpu process begin
// static Metric metric("netmate.out", 10, 2);
//
// void pass_tuple_info(struct pkt_tuple tuple, struct pkt_info info){
//   metric.process_pkt(tuple, info);
// }
//
// void output_flow_features(){
//   metric.output_flow_features();
// }
//
// void clean_flow_stats(uint64_t timestamp){
//   metric.clean_flow_stats(timestamp);
// }
// cpu process end

//gpu process begin
static GPU_Metric metric;

void pass_cpu_pkts(struct pkt_tuple_info *cpu_pkts){
  metric.add_pkts(cpu_pkts);
}

void output_flow_features(){
  cout << "call output_flow_features" << endl;
  metric.gpu_output_stats();
}

void clean_flow_stats(uint64_t timestamp){
  cout << "call clean_flow_stats" << endl;
  metric.clean_flow_stats(timestamp);
}

void quit_gpu_processing(){
  metric.quit();
}

void gpu_process(){
  metric.process();
}
//gpu process end
