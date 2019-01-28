# iswitch
Feature_extraction is about how to get the feature from the pkts and flows received from NIC of your server.To run this part,you should first bind your NIC cards which receive pkts to DPDK(refer to the document of DPDK ).And just run  "sudo ./run.sh"
For detail configuration,just  run "vi run.sh ".There -p means how many port your NICs totally provid,-c means how many cores you are going to bind to this program.And remember to change ./dpdk_code/main.c and ./dpdk_code/packet_processor.c ,(about the core and the port)to consist with your configuration just did.
Any questions about how to run this.connect chensw@ustc.edu.cn.
