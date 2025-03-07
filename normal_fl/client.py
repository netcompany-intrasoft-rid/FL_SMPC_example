import argparse
import logging
import threading
from typing import List, Dict, Tuple
import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from utils import partition_dataset, load_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NormalFLClient(fl.client.NumPyClient):
    def __init__(self, model: 'tf.keras.Model', num_clients: int, client_id: int):
        self.model = model
        self.num_clients = num_clients
        self.client_id = client_id
        self.model_shape = [layer.shape for layer in self.model.get_weights()]
        self.shutdown_flag = threading.Event()

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

        self.x_train, self.y_train = partition_dataset(x_train, y_train, self.num_clients, self.client_id)
        self.x_test, self.y_test = partition_dataset(x_test, y_test, self.num_clients, self.client_id)


    def get_parameters(self):
        logger.info(f"Client {self.client_id}: get_parameters() called")
        return self.model.get_weights()

    def fit(self,parameters: List[np.ndarray], config: Dict)  -> Tuple[List[np.ndarray], int, Dict]:
        logger.info(f"Client {self.client_id}: fit() called")
        processed_parameters = parameters.copy()

        self.model.set_weights(processed_parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=16, verbose=1)

        return self.model.get_weights(), len(self.x_train), {}


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

    def start(self):
        fl.client.start_client(server_address="localhost:8080", client=self.to_client())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPC Client with Hybrid Discovery")
    parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    args = parser.parse_args()
    # Load the initial model
    initial_model = load_model()

    # Create and start the client
    fl_client = NormalFLClient(model=initial_model, num_clients=args.num_clients, client_id=args.client_id)

    try:
        fl_client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")