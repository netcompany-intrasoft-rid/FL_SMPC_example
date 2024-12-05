import threading
from typing import List, Tuple, Optional, Dict
import flwr as fl
import numpy as np
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from utils import PRIME, SCALE_FACTOR, sparse_parameters_to_ndarrays, ndarrays_to_sparse_parameters

class SMPCServer(fl.server.strategy.FedAvg):
    def __init__(self, num_clients: int, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0):
        super().__init__()
        self.num_clients = num_clients
        self.clients = []
        self.wait_timeout = 60 # seconds
        self.clients_ready_event = threading.Event()
        self.model_structure = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Ensure the server waits for all clients to connect before initializing parameters"""
        print(f"Waiting for {self.num_clients} clients to connect...")
        client_manager.wait_for(num_clients=self.num_clients, timeout=self.wait_timeout)
        self.clients_ready_event.set()
        print("All clients connected. Initializing parameters...")
        return fl.common.ndarrays_to_parameters(self.generate_initial_weights())

    def generate_initial_weights(self) -> List[np.ndarray]:
        """Generate initial weights for the model"""
        np.random.seed(42) # for reprodcibility

        # Layer 1: Dense 128
        weights1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2 / 784)
        bias1 = np.zeros(128, dtype=np.float32)

        # Layer 2: Dense 10
        weights2 = np.random.randn(128, 10).astype(np.float32) * np.sqrt(2 / 128)
        bias2 = np.zeros(10, dtype=np.float32)

        return [weights1, bias1, weights2, bias2]

    def fixed_to_float(self, fixed_point: np.ndarray, scale_factor: float = 1e6) -> np.ndarray:
        """Convert fixed point to float"""
        return (fixed_point.astype(np.float32) / scale_factor) - (PRIME / (2 * scale_factor))

    def aggregate_smpc_weights(self, weights_results: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Aggregate weights using Secure Multi-Party Computation (SMPC)"""
        aggregated_weights = []

        for layer_weight in zip(**weights_results):
            layer_shape = layer_weight[0].shape
            flattened_weights = [w.flatten() for w in layer_weight]

            # Sum up the shares
            aggregated_layers = np.zeros_like(flattened_weights[0])
            for weight in flattened_weights:
                aggregated_layers  = (aggregated_layers + weight) % PRIME

            # Convert back to float and reshape
            aggregated_layer = self.fixed_to_float(aggregated_layers)
            aggregated_layer = aggregated_layer.reshape(layer_shape)
            aggregated_weights.append(aggregated_layer)

        return aggregated_weights

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """Aggregate the results of the clients' fit using SMPC protocol"""
        if not results:
            return None, {}

        # extract weights and metrics
        weights_results = [sparse_parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Aggregate weights using SMPC
        aggregated_weights = self.aggregate_smpc_weights(weights_results)

        # Convert aggregated weights to Parameters
        aggregated_parameters = ndarrays_to_sparse_parameters(aggregated_weights)

        # Aggregate metrics (if any)
        metrics_aggregated = {}
        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the results of the clients' evaluation using SMPC protocol"""
        if not results:
            print(f"Round {server_round}: No successfull evaluations. Failures: {failures}")
            return None, {}

        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        losses = [evaluate_res.metrics["loss"] for _, evaluate_res in results]
        num_samples = [evaluate_res.num_examples for _, evaluate_res in results]

        # Calculate weighted average of accuracies and losses
        weighted_accuracy = np.average(accuracies, weights=num_samples)
        weighted_loss = np.average(losses, weights=num_samples)

        print(f"Round {server_round}: Weighted Accuracy: {weighted_accuracy}, Weighted Loss: {weighted_loss}")
        return weighted_loss, {"accuracy": weighted_accuracy}

    def configure_fit(self, server_round: int,
                      parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Wait for all clients to connect before configuring fit"""
        print(f"Round {server_round}: Waiting for all clients to connect...")

        if server_round == 1:
            # in the fits round, we wait for all clients to connect
            if not self.clients_ready_event.wait(self.wait_timeout):
                print(f"Warning: Not all clients connected after {self.wait_timeout} seconds")

        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return fit_ins

if __name__ == "__main__":
    strategy = SMPCServer(num_clients=3, fraction_fit=1.0, fraction_evaluate=1.0)
    fl.server.start_server(server_address="localhost:8080",
                           strategy=strategy,
                           config=fl.server.ServerConfig(num_rounds=10))
