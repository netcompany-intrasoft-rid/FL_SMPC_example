import argparse
import logging
import threading
import time
from concurrent import futures
import flwr as fl
import grpc
import numpy as np
from tensorflow.keras.datasets import mnist

import smpc_pb2
import smpc_pb2_grpc
from utils import partition_dataset, load_model, ndarrays_to_sparse_parameters, fixed_to_float, float_to_fixed_point

PRIME = 2**31 - 1
SCALE_FACTOR = 1e6

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SMPCServicer(smpc_pb2_grpc.SMPCServicer):
    def __init__(self, client):
        self.client = client

    def SendShares(self, request, context):
        client_id = request.client_id
        shares = []
        for share in request.shares:
            shape = tuple(share.shape)
            data = np.frombuffer(share.data)
            if data.size != np.prod(shape):
                logger.error(f"Mismatch in data size and shape: {data.size} != {np.prod(shape)}")
                return smpc_pb2.AckResponse(status="ERROR")
            shares.append(data.reshape(shape))
        self.client.receive_shares(client_id, shares)
        return smpc_pb2.AckResponse(status="ACK")

class SMPCClient(fl.client.NumPyClient):
    def __init__(self, model, client_id, client_port, peer_addresses):
        self.model = model
        self.client_id = client_id
        self.peer_addresses = peer_addresses
        self.client_port = client_port
        self.received_shares = {}
        self.num_clients = len(peer_addresses) + 1
        self.all_shares_received = threading.Event()
        self.peer_stubs = {}
        self.all_peers_connected = threading.Event()
        self.model_shape = [layer.shape for layer in self.model.get_weights()]
        self.shutdown_flag = threading.Event()
        self.own_shares = None

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

        self.x_train, self.y_train = partition_dataset(x_train, y_train, self.num_clients, self.client_id)
        self.x_test, self.y_test = partition_dataset(x_test, y_test, self.num_clients, self.client_id)

        # start server listening to the rest of clients for shares
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        smpc_pb2_grpc.add_SMPCServicer_to_server(SMPCServicer(self), self.server)
        self.server.add_insecure_port(f"[::]:{self.client_port}")
        self.server.start()

    def cleanup(self):
        self.server.stop(0)
        self.shutdown_flag.set()

    def connect_to_peers(self):
        logger.info(f"Client {self.client_id} connecting to peers")
        for peer_id, peer_address in enumerate(self.peer_addresses):
            try:
                channel = grpc.insecure_channel(peer_address)
                self.peer_stubs[peer_id] = smpc_pb2_grpc.SMPCStub(channel)
                logger.info(f"Client {self.client_id} connected to peer {peer_id} at {peer_address}")
            except Exception as e:
                logger.error(f"Client {self.client_id} failed to connect to peer {peer_id}: {e}")

        if len(self.peer_stubs) == len(self.peer_addresses):
            logger.info(f"Client {self.client_id} connected to all peers")
            self.all_peers_connected.set()
        else:
            logger.warning(f"Client {self.client_id} failed to connect to all peers")

    def get_parameters(self):
        logger.info(f"Client {self.client_id}: get_parameters() called")
        return self.model.get_weights()
    
    def preprocess_incoming_parameters(self, parameters):
        model_weights = []
        current_idx = 0

        if isinstance(parameters, np.lib.npyio.NpzFile):
            for expected_shape in self.model_shape:
                total_elements = np.prod(expected_shape)

                for key in parameters.files:
                    param_data = parameters[key]
                    logger.info(f"Extracting {key} from npz file: shape={param_data.shape}")
                    # Convert npz array data to the correct shape
                    if param_data.size >= total_elements:
                        reshaped_param = param_data[:total_elements].reshape(expected_shape)
                        model_weights.append(reshaped_param)
                        break
        elif isinstance(parameters, list):
            # handle list of arrays case
            for expected_shape in self.model_shape:
                total_elements = np.prod(expected_shape)
                if current_idx < len(parameters):
                    param = parameters[current_idx]

                    if isinstance(param, np.lib.npyio.NpzFile):
                        for key in param.files:
                            param_data = param[key]
                            logger.info(f"Extracting {key} from npz file: shape={param_data.shape}")
                            flattened_param = param_data.flatten()
                            logger.info(f"Flattened size={flattened_param.size}")
                            if flattened_param.size >= total_elements:
                                reshaped_param = flattened_param[:total_elements].reshape(expected_shape)
                                model_weights.append(reshaped_param)
                                break
                    else:
                        flattened_param = np.array(param).flatten()
                        logger.info(f"Parameter {current_idx}: shape={param.shape}, flattened size={flattened_param.size}")
                        if flattened_param.size >= total_elements:
                            reshaped_param = flattened_param[:total_elements].reshape(expected_shape)
                            model_weights.append(reshaped_param)
                    current_idx += 1
        return model_weights


    def fit(self,parameters, config):
        logger.info(f"Client {self.client_id}: fit() called")
        processed_parameters = self.preprocess_incoming_parameters(parameters)
        print("processed parameters shapes")
        print([param.shape for param in processed_parameters])

        self.model.set_weights(processed_parameters)
        print("Train size:")
        print(len(self.x_train))
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=1)

        secret_shares = self.create_secret_shares_new(self.model, self.num_clients)
        self.own_shares = secret_shares[self.client_id]
        # fixed_point_shares = {
        #     client_id: [
        #         np.array([float_to_fixed_point(w) for w in matrix.flatten()]).reshape(matrix.shape) for matrix in shares]
        #         for client_id, shares in secret_shares.items() if client_id != self.client_id
        # }

        self.send_shares_to_peers(secret_shares)

        if not self.all_shares_received.wait(timeout=60):
            logger.error(f"Not all shares received for client {self.client_id}")

        aggregated_shares = self.aggregate_received_shares_new()
        self.all_shares_received.clear()
        self.received_shares.clear()
        return aggregated_shares, len(self.x_train), {}

    def evaluate(self, parameters, config):
        logger.info(f"Client {self.client_id}: evaluate() called")

        # Convert parameters to the correct shape
        model_weights = self.preprocess_incoming_parameters(parameters)

        model_weights_shape = [weight.shape for weight in model_weights]

        if model_weights_shape != self.model_shape:
            logger.error(f"Model weights mismatch: {model_weights_shape} != {self.model_shape}")
            return 0.0, len(self.x_test), {"accuracy": 0.0}

        for idx, (weight, expected_shape) in enumerate(zip(model_weights, self.model_shape)):
            if weight.shape != expected_shape:
                logger.error(f"Weight shape mismatch at index {idx}: {weight.shape} != {expected_shape}")
                return 0.0, len(self.x_test), {"accuracy": 0.0}

        logger.info("Successfully reshaped weights to match model architecture")
        self.model.set_weights(model_weights)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        logger.info(f"Client {self.client_id} evaluated model with loss: {loss} and accuracy: {accuracy}")
        return loss, len(self.x_test), {"accuracy": accuracy}

    def create_secret_shares_new(self, fl_model, num_clients):
        weights = fl_model.get_weights()
        secret_shares = {i: [] for i in range(num_clients)}

        for weight_matrix in weights:
            split_matrices = [[] for _ in range(num_clients)]

            for w in np.nditer(weight_matrix, order='C'):
                shares = [np.random.uniform(-1e12, 1e12) for _ in range(num_clients - 1)]
                last_share = (w - sum(shares))
                shares.append(last_share)
                for i in range(num_clients):
                    split_matrices[i].append(shares[i])

            for i in range(num_clients):
                reshaped_matrix = np.array(split_matrices[i]).reshape(weight_matrix.shape)
                secret_shares[i].append(reshaped_matrix)

        return secret_shares

    def send_shares_to_peers(self, secret_shares):
        for peer_id, stub in self.peer_stubs.items():
            shares = list(secret_shares.values())[peer_id]
            share_protos = []
            for matrix in shares:
                flattened_data = matrix.flatten()
                share_protos.append(smpc_pb2.Share(data=flattened_data.tobytes(), shape=list(matrix.shape))) # Convert shape to list
            request = smpc_pb2.SharesRequest(client_id=self.client_id, shares=share_protos)
            try:
                response = stub.SendShares(request)
                if response.status == "ACK":
                    logger.info(f"Client {self.client_id} sent shares to peer {peer_id}")
                else:
                    logger.error(f"Unexpected response from client {peer_id}: {response.status}")
            except grpc.RpcError as e:
                logger.error(f"Client {self.client_id} failed to send shares to peer {peer_id}: {e}")

    def aggregate_received_shares(self):
        logger.info(f"Client {self.client_id} started aggregating received shares")
        aggregated_weights = []
        all_shares = [self.own_shares] + list(self.received_shares.values())

        for weight_matrix_list in zip(*all_shares):
            aggregated_matrix = np.zeros_like(weight_matrix_list[0])
            for weight_matrix in weight_matrix_list:
                if aggregated_matrix.shape != weight_matrix.shape:
                    aggregated_matrix = aggregated_matrix.reshape(weight_matrix.shape)
                aggregated_matrix = (aggregated_matrix + weight_matrix) % PRIME
            aggregated_matrix = np.array([fixed_to_float(x) for x in aggregated_matrix.flatten()]).reshape(aggregated_matrix.shape)
            aggregated_weights.append(aggregated_matrix)

        return aggregated_weights
    
    def aggregate_received_shares_new(self):
        logger.info(f"Client {self.client_id} started aggregating received shares")
        aggregated_weights = []
        all_shares = [self.own_shares] + list(self.received_shares.values())

        for weight_matrix_list in zip(*all_shares):
            aggregated_matrix = np.sum(weight_matrix_list, axis=0)  # No modulo operation
            aggregated_weights.append(aggregated_matrix)

        return aggregated_weights


    def receive_shares(self, client_id, shares):
        logger.info(f"Client {self.client_id} received shares from client {client_id}")
        self.received_shares[client_id] = shares
        if len(self.received_shares) == len(self.peer_addresses):
            logger.info(f"Client {self.client_id} received all shares")
            self.all_shares_received.set()

    def start(self):
        time.sleep(2)
        self.connect_to_peers()

        logger.info(f"Client {self.client_id} waiting for all peers to connect")
        self.all_peers_connected.wait()
        logger.info("All peer connections established. Connecting to Flower server...")
        fl.client.start_client(server_address="localhost:8080", client=self.to_client())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPC Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--client_port", type=int, required=True, help="Client port")
    parser.add_argument("--peer_addresses", type=str, required=True, help="Peer addresses")
    args = parser.parse_args()
    peer_addresses = args.peer_addresses.split(",")

    model = load_model()
    fl_client = SMPCClient(model, args.client_id, args.client_port, peer_addresses)
    try:
        fl_client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        fl_client.cleanup()