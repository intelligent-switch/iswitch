# iswitch
Dependencies  
Ubuntu 16.04  
gcc >= 4.8  
Python 3.6  
Pytorch 0.4.0  
make  
DPDK 18.0  
CUDA 8.0  

# Directory
Dataset: We collected the metadata of the traffic from our campus network as the training and testing datasets. The dataset is composed of 20 popular applications: WeChat, BitTorrent, Skype, video streaming and online games, etc.   
  
Feature_extraction:  High-speed traffic feature extraction with DPDK. The parameter "-p" specifies the number of ports the NICs provided, "-c" specifies the number of NIC queues. For more detail, please see "run.sh".  

application identification: We implement  specifc application identifcation methods into iSwitch.   
anomaly detection: The iSwitch well supports high throughput online anomaly detection and gains high detection accuracy.  



Note: To run this process, you need to bind NICs to DPDK and execute the script "./run.sh".   








