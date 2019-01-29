obj = dpdk_code/build/main.o dpdk_code/build/packet_processor.o

ccsrc = $(wildcard *.cpp)
cusrc = $(wildcard *.cu)
obj += $(ccsrc:.cpp=.o) $(cusrc:.cu=.o)

CXX = g++

CPPFLAGS = -std=c++11 -I/usr/local/cuda/include

%.o: %.cu
	nvcc -std=c++11 -I/usr/local/cuda/samples/common/inc/ -o $@ -c $<


LDFLAGS = -L$(RTE_SDK)/$(RTE_TARGET)/lib -lrte_flow_classify -lrte_pipeline -lrte_table -lrte_port -lrte_pdump -lrte_distributor -lrte_ip_frag -lrte_gro -lrte_gso -lrte_meter -lrte_lpm -Wl,--whole-archive -lrte_acl -Wl,--no-whole-archive -lrte_jobstats -lrte_metrics -lrte_bitratestats -lrte_latencystats -lrte_power -lrte_timer -lrte_efd -Wl,--whole-archive -lrte_cfgfile -lrte_hash -lrte_member -lrte_vhost -lrte_kvargs -lrte_mbuf -lrte_net -lrte_ethdev -lrte_bbdev -lrte_cryptodev -lrte_security -lrte_eventdev -lrte_rawdev -lrte_mempool -lrte_mempool_ring -lrte_ring -lrte_pci -lrte_eal -lrte_cmdline -lrte_reorder -lrte_sched -lrte_kni -lrte_bus_pci -lrte_bus_vdev -lrte_mempool_stack -lrte_pmd_af_packet -lrte_pmd_ark -lrte_pmd_avf -lrte_pmd_avp -lrte_pmd_bnxt -lrte_pmd_bond -lrte_pmd_cxgbe -lrte_pmd_e1000 -lrte_pmd_ena -lrte_pmd_enic -lrte_pmd_fm10k -lrte_pmd_failsafe -lrte_pmd_i40e -lrte_pmd_ixgbe -lrte_pmd_kni -lrte_pmd_lio -lrte_pmd_nfp -lrte_pmd_null -lrte_pmd_qede -lrte_pmd_ring -lrte_pmd_sfc_efx -lrte_pmd_tap -lrte_pmd_thunderx_nicvf -lrte_pmd_vdev_netvsc  -lrte_pmd_virtio -lrte_pmd_vhost -lrte_pmd_vmxnet3_uio -lrte_pmd_bbdev_null -lrte_pmd_null_crypto -lrte_pmd_crypto_scheduler -lrte_pmd_skeleton_event -lrte_pmd_sw_event -lrte_pmd_octeontx_ssovf -lrte_mempool_octeontx -lrte_pmd_octeontx -lrte_pmd_opdl_event -lrte_pmd_skeleton_rawdev -Wl,--no-whole-archive -lrt -lnuma -lm -lrt -lm -lm -lnuma -lpthread -ldl

LDFLAGS += -L/usr/local/cuda/lib64  -lcudart

netmate: $(obj)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) netmate
