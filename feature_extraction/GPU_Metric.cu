#include "GPU_Metric.h"
#include "Hash_Utils.h"
#include "GPU_Hash.h"

#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>

std::ostream &operator<<(std::ostream &os, const pkt_tuple &tuple);

std::ostream &operator<<(std::ostream &os, const flow_features &flow);

std::ostream &operator<<(std::ostream &os, const features &bidirec_flow);

__device__ void init_new_entry(struct features *bidirec_flow, struct pkt_tuple *tuple, struct pkt_info *info){
  switch(tuple->proto){
    case 6: bidirec_flow->hlen = UDP_HEADER_LEN; break;
    case 17: bidirec_flow->hlen = TCP_HEADER_LEN; break;
  }

  bidirec_flow->first_timestamp = info->timestamp;
  bidirec_flow->last_timestamp = info->timestamp;
  bidirec_flow->active_timestamp = info->timestamp;

  bidirec_flow->forward.timestamp = info->timestamp;
  bidirec_flow->forward.total_packets = 1;
  bidirec_flow->forward.total_volume = info->data_len;
  bidirec_flow->forward.min_pktl = info->data_len;
  bidirec_flow->forward.max_pktl = info->data_len;
  bidirec_flow->forward.sqsum_pktl = info->data_len * info->data_len;

  if(info->psh) ++bidirec_flow->forward.psh_cnt;
  if(info->urg) ++bidirec_flow->forward.urg_cnt;
  bidirec_flow->forward.total_hlen = bidirec_flow->hlen;
}

__device__ void update_entry(hash_entry *entry, struct pkt_tuple *tuple, struct pkt_info *info){
  int diff = info->timestamp - entry->bidirec_flow.last_timestamp;
  if(diff > IDLE_THRESHOLD){
    int cur_active = entry->bidirec_flow.last_timestamp - entry->bidirec_flow.active_timestamp;
    if(cur_active > 0){
      if(cur_active < entry->bidirec_flow.min_active || entry->bidirec_flow.min_active == 0) entry->bidirec_flow.min_active = cur_active;
      if(cur_active > entry->bidirec_flow.max_active) entry->bidirec_flow.max_active = cur_active;
      entry->bidirec_flow.sum_active += cur_active;
      entry->bidirec_flow.sqsum_active += cur_active * cur_active;
      ++entry->bidirec_flow.active_times;
    }
    if(diff < entry->bidirec_flow.min_idle || entry->bidirec_flow.min_idle == 0) entry->bidirec_flow.min_idle = diff;
    if(diff > entry->bidirec_flow.max_idle) entry->bidirec_flow.max_idle = diff;
    entry->bidirec_flow.sum_idle += diff;
    entry->bidirec_flow.sqsum_idle += diff * diff;
    ++entry->bidirec_flow.idle_times;
    entry->bidirec_flow.active_timestamp = info->timestamp;
  }
  entry->bidirec_flow.last_timestamp = info->timestamp;

  struct flow_features *flow;
  if(tuple->src_ip == entry->tuple.src_ip) flow = &(entry->bidirec_flow.forward);
  else flow = &(entry->bidirec_flow.backward);

  flow->total_hlen += entry->bidirec_flow.hlen;
  if(info->psh) ++flow->psh_cnt;
  if(info->urg) ++flow->urg_cnt;

  if(flow->total_packets == 0){
    flow->timestamp = info->timestamp;
    flow->total_packets = 1;
    flow->total_volume = info->data_len;
    flow->min_pktl = info->data_len;
    flow->max_pktl = info->data_len;
    flow->sqsum_pktl = info->data_len * info->data_len;
    return;
  }

  ++flow->total_packets;
  flow->total_volume += info->data_len;
  if(flow->min_pktl > info->data_len) flow->min_pktl = info->data_len;
  if(flow->max_pktl < info->data_len) flow->max_pktl = info->data_len;
  flow->sqsum_pktl += info->data_len * info->data_len;

  int interval = info->timestamp - flow->timestamp;
  flow->timestamp = info->timestamp;
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

__global__ void gpu_extract_features(struct pkt_tuple_info *pkts, int *first_index, hash_entry *hash_table){
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int cur_index = first_index[index];
  int num = 0;
  hash_entry *entry;

  while(cur_index != -1){
    entry = find_entry(hash_table, HASH_ENTRIES, pkts[cur_index].hash, pkts[cur_index].tuple);
    if(entry == NULL){
      struct features bidirec_flow = {0};
      init_new_entry(&bidirec_flow, &(pkts[cur_index].tuple), &(pkts[cur_index].info));
      insert_entry(hash_table, HASH_ENTRIES, pkts[cur_index].hash, pkts[cur_index].tuple, bidirec_flow);
    }
    else update_entry(entry, &(pkts[cur_index].tuple), &(pkts[cur_index].info));
    ++num;
    cur_index = pkts[cur_index].next_index;
  }

  //printf("thread%d process %d packets\n", index, num);
  //printf("thread%d process %dth packet\n", index, cur_index);
}

__global__ void gpu_update_hash_table(hash_entry *src_table, hash_entry *dst_table, uint64_t update_timestamp){
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int N = HASH_ENTRIES / TotalThreads + 1;
  int start = index * N;
  int end = start + N;
  if(end > HASH_ENTRIES) end = HASH_ENTRIES;

  for(int i = start; i < end; ++i){
    if(src_table[i].state != EMPTY_ENTRY && update_timestamp - src_table[i].bidirec_flow.last_timestamp < TIMEOUT_THRESHOLD){
      insert_entry(dst_table, HASH_ENTRIES, src_table[i].hash, src_table[i].tuple, src_table[i].bidirec_flow);
    }
  }
}

GPU_Metric::GPU_Metric(): force_quit(false), array_num(0), max_array_num(0), has_data(false), copy_to_cpu(false), update_hash_table(false) {
  gpu_hash_table[0] = create_hash_table(HASH_ENTRIES);
  gpu_hash_table[1] = create_hash_table(HASH_ENTRIES);
  hash_table = gpu_hash_table[0];
  cpu_hash_table = new hash_entry[HASH_ENTRIES];

  cpu_first_index = new int[TotalThreads];
  cpu_last_index = new int[TotalThreads];

  memset(cpu_first_index, -1, TotalThreads * sizeof(int));
  memset(cpu_last_index, -1, TotalThreads * sizeof(int));

  cudaMalloc(&gpu_pkts, ArraySize * sizeof(struct pkt_tuple_info));
  cudaMalloc(&gpu_first_index, TotalThreads * sizeof(int));

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = INADDR_ANY;
}

GPU_Metric::~GPU_Metric(){
  free_hash_table(gpu_hash_table[0]);
  free_hash_table(gpu_hash_table[1]);
  delete []cpu_hash_table;

  delete []cpu_first_index;
  delete []cpu_last_index;

  cudaFree(gpu_pkts);
  cudaFree(gpu_first_index);

  close(sockfd);
}

void GPU_Metric::process(){
  struct pkt_tuple_info *pkts;

  while(!force_quit){

    if(has_data){
      q_m.lock();
      pkts = q.front();
      q.pop();
      if(array_num > max_array_num) max_array_num = array_num;
      --array_num;
      if(array_num == 0) has_data = false;
      q_m.unlock();

      process_pkts(pkts);
    }

  }
}

void GPU_Metric::process_pkts(struct pkt_tuple_info *pkts){
  int index;

  for(int i = 0; i < ArraySize; ++i){
    index = pkts[i].hash % TotalThreads;
    if(cpu_first_index[index] == -1){
      cpu_first_index[index] = i;
      cpu_last_index[index] = i;
    }
    else{
      pkts[cpu_last_index[index]].next_index = i;
      cpu_last_index[index] = i;
    }
    
  }

  if(update_hash_table){
    hash_entry *pre_hash_table = hash_table;
    if(hash_table == gpu_hash_table[0]){
      hash_table = gpu_hash_table[1];
    }
    else{
      hash_table = gpu_hash_table[0];
    }
    gpu_update_hash_table<<<numBlocks, threadsPerBlock>>>(pre_hash_table, hash_table, update_timestamp);
    cudaMemset(pre_hash_table, 0, HASH_ENTRIES * sizeof(hash_entry));
    update_hash_table = false;
  }

  cudaMemcpy(gpu_pkts, pkts, ArraySize * sizeof(struct pkt_tuple_info), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_first_index, cpu_first_index, TotalThreads * sizeof(int), cudaMemcpyHostToDevice);

  gpu_extract_features<<<numBlocks, threadsPerBlock>>>(gpu_pkts, gpu_first_index, hash_table);

  memset(cpu_first_index, -1, TotalThreads * sizeof(int));
  memset(cpu_last_index, -1, TotalThreads * sizeof(int));

  free(pkts);

  if(copy_to_cpu){
    cudaMemcpy(cpu_hash_table, hash_table, HASH_ENTRIES * sizeof(hash_entry), cudaMemcpyDeviceToHost);
    copy_to_cpu = false;
  }
}

void GPU_Metric::clean_flow_stats(uint64_t timestamp){
  update_timestamp = timestamp;
  update_hash_table = true;
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff;
  while(update_hash_table){
    end = std::chrono::system_clock::now();
    diff = end - start;
    if(diff.count() > 2){
      std::cout << "do not need to update hash table" << std::endl;
      update_hash_table = false;
      return;
    }
  }
  end = std::chrono::system_clock::now();
  diff = end - start;
  std::cout << "updated hash table, used " << diff.count() << "s!" << std::endl;
}

void GPU_Metric::gpu_output_stats(){
  std::cout << "max_array_num = " << max_array_num << std::endl;

  copy_to_cpu = true;
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff;
  while(copy_to_cpu){
    end = std::chrono::system_clock::now();
    diff = end - start;
    if(diff.count() > 1){
      std::cout << "no features to record" << std::endl;
      copy_to_cpu = false;
      return;
    }
  }
  std::cout << "copied hash_table to cpu_hash_table" << std::endl;
  //output_to_file();
  output_through_udp();
}

void GPU_Metric::output_to_file(){
  std::ofstream ofs("gpu_out", std::ofstream::app);
  ofs << "Output begin!" << std::endl;
  auto start = std::chrono::system_clock::now();
  for(int i = 0; i < HASH_ENTRIES; ++i){
    if(cpu_hash_table[i].state != 0){
      ofs << cpu_hash_table[i].tuple << " " << cpu_hash_table[i].bidirec_flow << std::endl;
    }
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - start;
  ofs << "Output end, used " << diff.count() << "s!" << std::endl;
  ofs.close();
}

void GPU_Metric::output_through_udp(){
  //auto start = std::chrono::system_clock::now();

  int index = 0;
  while(index < HASH_ENTRIES){
    std::ostringstream oss;
    while(oss.tellp() < 1470 && index < HASH_ENTRIES){
      if(cpu_hash_table[index].state != 0){
        oss << cpu_hash_table[index].tuple << " " << cpu_hash_table[index].bidirec_flow << std::endl;
      }
      ++index;
    }
    std::string msg = oss.str();
    sendto(sockfd, msg.c_str(), msg.size(), MSG_CONFIRM, (const struct sockaddr *)&servaddr, sizeof(servaddr));
  }

  // auto end = std::chrono::system_clock::now();
  // std::chrono::duration<double> diff = end - start;
  // std::string msg("Output end, used ");
  // msg += std::to_string(diff.count());
  // msg += "\n";
  // sendto(sockfd, msg.c_str(), msg.size(), MSG_CONFIRM, (const struct sockaddr *)&servaddr, sizeof(servaddr));
}
