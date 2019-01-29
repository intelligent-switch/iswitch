#ifndef _GPU_HASH_H
#define _GPU_HASH_H

#include <stdlib.h>
#include <cuda_runtime.h>

#include "data_types.h"

#define EMPTY_ENTRY 0
#define NON_EMPTY_ENTRY 1

hash_entry *create_hash_table(int cap){
  hash_entry *ret;
  cudaMalloc(&ret, cap * sizeof(hash_entry));
  cudaMemset(ret, 0, cap * sizeof(hash_entry));
  return ret;
}

void free_hash_table(hash_entry *table){
  cudaFree(table);
}

__device__ bool is_equal(struct pkt_tuple t1, struct pkt_tuple t2){
  if(t1.src_ip == t2.src_ip && t1.dst_ip == t2.dst_ip && t1.proto == t2.proto
    && t1.src_port == t2.src_port && t1.dst_port == t2.dst_port) return true;
  if(t1.src_ip == t2.dst_ip && t1.dst_ip == t2.src_ip && t1.proto == t2.proto
    && t1.src_port == t2.dst_port && t1.dst_port == t2.src_port) return true;
  return false;
}

__device__ hash_entry *find_entry(hash_entry *table, int cap, uint32_t hash, struct pkt_tuple tuple){
  int index = hash % cap;
  while(true){
    //while(atomicCAS(&(table[index].mutex), 0, 1) != 0);//lock table[index]
    if(table[index].state == EMPTY_ENTRY){
      //atomicExch(&(table[index].mutex), 0); //unlock table[index]
      return NULL;
    }

    //atomicExch(&(table[index].mutex), 0); //unlock table[index]
    if(is_equal(table[index].tuple, tuple)){
      return &(table[index]);
    }

    ++index;
    if(index == cap) index = 0;
  }
}

__device__ void insert_entry(hash_entry *table, int cap, uint32_t hash, struct pkt_tuple tuple, struct features bidirec_flow){
  int index = hash % cap;
  while(true){
    while(atomicCAS(&(table[index].mutex), 0, 1) != 0);//lock table[index]
    if(table[index].state == EMPTY_ENTRY){
      table[index].state = NON_EMPTY_ENTRY;
      atomicExch(&(table[index].mutex), 0); //unlock table[index]
      table[index].hash = hash;
      table[index].tuple = tuple;
      table[index].bidirec_flow = bidirec_flow;
      return;
    }
    else{
      atomicExch(&(table[index].mutex), 0); //unlock table[index]
      ++index;
      if(index == cap) index = 0;
    }
  }
}

#endif
