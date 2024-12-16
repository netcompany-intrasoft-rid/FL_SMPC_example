import argparse
import logging
import threading
import time
from typing import List, Dict, Tuple
from concurrent import futures
import flwr as fl
import grpc
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import smpc_pb2
import smpc_pb2_grpc
from utils import partition_dataset, load_model

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
            data = np.frombuffer(share.data, dtype=np.float32)
            if data.size != np.prod(shape):
                logger.error(f"Mismatch in data size and shape: {data.size} != {np.prod(shape)}")
                return smpc_pb2.AckResponse(status="ERROR")
            shares.append(data.reshape(shape))
        self.client.receive_shares(client_id, shares)
        return smpc_pb2.AckResponse(status="ACK")

class SMPCClient(fl.client.NumPyClient):
    def __init__(self, model: 'tf.keras.Model', client_id: int, client_port: int, peer_addresses: List[str]):
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
        try:
            self.server.stop(grace=5)
        except Exception as e:
            logger.error(f"An error occurred during cleanup: {e}")
        finally:
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

    def fit(self,parameters: List[np.ndarray], config: Dict)  -> Tuple[List[np.ndarray], int, Dict]:
        logger.info(f"Client {self.client_id}: fit() called")
        processed_parameters = parameters.copy()

        self.model.set_weights(processed_parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=1)

        secret_shares = list(self.split_model_weights_to_shares(self.model.get_weights(), self.num_clients).values())

        self.own_shares = secret_shares[self.client_id]

        secret_shares.pop(self.client_id)
        self.send_shares_to_peers(secret_shares)

        try:
            if not self.all_shares_received.wait(timeout=60):
                raise TimeoutError(f"Not all shares received for client {self.client_id}")
        except TimeoutError as e:
            logger.error(f"Client {self.client_id} failed to receive all shares: {e}")
            return None, 0, {}

        all_shares = [self.own_shares] + list(self.received_shares.values())
        aggregated_shares = self.reconstruct_model_weights(all_shares)
        self.all_shares_received.clear()
        self.received_shares.clear()

        return aggregated_shares, len(self.x_train), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict[str, float]]:
        logger.info(f"Client {self.client_id}: evaluate() called")

        model_weights = parameters.copy()
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

    def generate_value_secret_share(self, value: float, num_shares: int = 3) -> List[np.ndarray]:
        """
        Generates secret shares of a given value using additive secret sharing.

        This method splits a value into `num_shares` parts, where the sum of all shares equals the original value.
        The shares are randomly generated, except for the last share, which is determined so that the sum of 
        all shares equals the original value.

        Parameters:
        value (float): The value to be secret-shared.
        num_shares (int): The number of shares to split the value into. Default is 3.

        Returns:
        list: A list of secret shares (floating-point numbers).
        """
        shares = [np.random.uniform(-1, 1) for _ in range(num_shares - 1)]
        last_share = value - sum(shares)
        shares.append(last_share)
        return shares

    def generate_matrix_secret_shares(self, matrix: np.ndarray, num_shares: int = 3) -> List[np.ndarray]:
        """
        Generates secret shares for each element in a matrix using additive secret sharing.

        This method splits each element in the provided matrix into `num_shares` parts, where the sum of 
        all shares for each element equals the original value. The shares for each element are calculated 
        using the `generate_value_secret_share` method, and the result is a matrix of secret shares where 
        each element in the matrix has been split into shares.

        Parameters:
        matrix (numpy.ndarray): The matrix whose elements are to be secret-shared.
        num_shares (int): The number of shares to split each element into. Default is 3.

        Returns:
        list: A list of matrices, each containing one set of secret shares for each element in the original matrix.
        """
        shares = [np.zeros_like(matrix) for _ in range(num_shares)]
        for index, value in np.ndenumerate(matrix):
            value_shares = self.generate_value_secret_share(value, num_shares)
            for i in range(num_shares):
                shares[i][index] = value_shares[i]
        return shares

    def split_model_weights_to_shares(self, weights: List[np.ndarray], num_clients: int) -> Dict[int, List[np.ndarray]]:
        """
        Splits model weights into secret shares for each client.

        This method takes a list of model weights and splits each weight into secret shares using the 
        `generate_matrix_secret_shares` method. The number of shares per weight corresponds to the number of 
        clients specified (`num_clients`). The secret shares for each client are grouped together, so that 
        each client will receive their respective shares of all the model weights.

        Parameters:
        weights (list of numpy.ndarray): The list of model weights to be split into secret shares.
        num_clients (int): The number of clients that will receive secret shares. 

        Returns:
        dict: A dictionary where each key corresponds to a client (from 0 to `num_clients - 1`), 
            and the value is a list of secret shares for each model weight for that client.
        """
        shares_per_weight = [self.generate_matrix_secret_shares(weight, num_clients) for weight in weights]
        shares_grouped_by_client = {i: [shares[i] for shares in shares_per_weight] for i in range(num_clients)}
        return shares_grouped_by_client

    def reconstruct_model_weights(self, shares_grouped_by_client: Dict[int, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Reconstructs model weights from the secret shares provided by each client.

        This method takes the secret shares grouped by client and reconstructs the original model weights
        by aggregating the shares for each weight tensor. The method assumes that the shares for each weight 
        tensor are additive and that the sum of the shares equals the original weight.

        Parameters:
        shares_grouped_by_client (dict): A dictionary where each key corresponds to a client (from 0 to 
                                        the number of clients - 1), and the value is a list of secret 
                                        shares for each model weight.

        Returns:
        list: A list of reconstructed model weights, where each element corresponds to a model weight 
            reconstructed from the aggregated shares.
        """
        num_weights = len(shares_grouped_by_client[0])
        # Aggregate shares for each weight tensor
        reconstructed_weights = [
            sum(client_shares[i] for client_shares in shares_grouped_by_client)
            for i in range(num_weights)
        ]
        return reconstructed_weights

    def send_shares_to_peers(self, secret_shares: Dict[int, List[np.ndarray]]):
        """
        Sends secret shares to all peer clients.

        This method takes the secret shares generated for each client and sends them to the corresponding 
        peers using gRPC. Each peer client receives the shares as flattened byte data along with the shape 
        of the matrix. The method logs the outcome of the send operation and handles any errors during 
        the communication.

        Parameters:
        secret_shares (dict): A dictionary where the key is the peer client ID and the value is a list of 
                            secret shares (matrices) that should be sent to that peer.
        """
        for peer_id, stub in self.peer_stubs.items():
            shares = secret_shares[peer_id]
            share_protos = []
            for matrix in shares:
                flattened_data = matrix.flatten()
                share_protos.append(smpc_pb2.Share(data=flattened_data.astype(np.float32).tobytes(), shape=list(matrix.shape))) # Convert shape to list
            request = smpc_pb2.SharesRequest(client_id=self.client_id, shares=share_protos)
            try:
                response = stub.SendShares(request)
                if response.status == "ACK":
                    logger.info(f"Client {self.client_id} sent shares to peer {peer_id}")
                else:
                    logger.error(f"Unexpected response from client {peer_id}: {response.status}")
            except grpc.RpcError as e:
                logger.error(f"Client {self.client_id} failed to send shares to peer {peer_id}: {e}")

    def receive_shares(self, client_id: int, shares: List[np.ndarray]):
        """
        Receives secret shares from a peer client.

        This method is called when a client receives shares from another client. It logs the receipt of shares 
        and stores them in the `received_shares` dictionary. Once all shares from peers are received, it triggers 
        the `all_shares_received` event.

        Parameters:
        client_id (int): The ID of the client sending the shares.
        shares (list): A list of secret shares received from the peer client.
        """
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
    peers = args.peers.split(",")

    model = load_model()
    fl_client = SMPCClient(model, args.client_id, args.client_port, peers)
    try:
        fl_client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        fl_client.cleanup()
