python client.py --num_clients=3 --client_id=0 &
python client.py --num_clients=3 --client_id=1 &
python client.py --num_clients=3 --client_id=2 &
wait # Ensures the script waits for all background processes to complete
