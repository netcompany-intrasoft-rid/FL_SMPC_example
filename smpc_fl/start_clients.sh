python client.py --client_id=0 --client_port=50051 --peer_addresses="localhost:50052,localhost:50053,localhost:50054,localhost:50055,localhost:50056,localhost:50057,localhost:50058" &
python client.py --client_id=1 --client_port=50052 --peer_addresses="localhost:50051,localhost:50053,localhost:50054,localhost:50055,localhost:50056,localhost:50057,localhost:50058" &
python client.py --client_id=2 --client_port=50053 --peer_addresses="localhost:50051,localhost:50052,localhost:50054,localhost:50055,localhost:50056,localhost:50057,localhost:50058" &
python client.py --client_id=3 --client_port=50054 --peer_addresses="localhost:50051,localhost:50052,localhost:50053,localhost:50055,localhost:50056,localhost:50057,localhost:50058" &
python client.py --client_id=4 --client_port=50055 --peer_addresses="localhost:50051,localhost:50052,localhost:50053,localhost:50054,localhost:50056,localhost:50057,localhost:50058" &
python client.py --client_id=5 --client_port=50056 --peer_addresses="localhost:50051,localhost:50052,localhost:50053,localhost:50054,localhost:50055,localhost:50057,localhost:50058" &
python client.py --client_id=6 --client_port=50057 --peer_addresses="localhost:50051,localhost:50052,localhost:50053,localhost:50054,localhost:50055,localhost:50056,localhost:50058" &
python client.py --client_id=7 --client_port=50058 --peer_addresses="localhost:50051,localhost:50052,localhost:50053,localhost:50054,localhost:50055,localhost:50056,localhost:50057" &
wait