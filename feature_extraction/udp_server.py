from socket import socket, AF_INET, SOCK_DGRAM

ADDR = '0.0.0.0'
PORT = 40000
flow_stats = open('flow_stats', 'w')

if __name__ == '__main__':
	s = socket(AF_INET, SOCK_DGRAM)
	s.bind((ADDR, PORT))

	while True:
		data, addr = s.recvfrom(2048)
		flow_stats.write(data.decode('utf-8'))
