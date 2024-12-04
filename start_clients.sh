python client.py --client_id=0 --client_port=50051 --peer_addresses="localhost:50052,localhost:50053"
python client.py --client_id=1 --client_port=50052 --peer_addresses="localhost:50051,localhost:50053"
python client.py --client_id=2 --client_port=50053 --peer_addresses="localhost:50051,localhost:50052"