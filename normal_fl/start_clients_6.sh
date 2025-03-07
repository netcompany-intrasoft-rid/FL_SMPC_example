python client.py --num_clients=6 --client_id=0 &
python client.py --num_clients=6 --client_id=1 &
python client.py --num_clients=6 --client_id=2 &
python client.py --num_clients=6 --client_id=3 &
python client.py --num_clients=6 --client_id=4 &
python client.py --num_clients=6 --client_id=5 &
wait # Ensures the script waits for all background processes to complete
