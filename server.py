import threading
from typing import List, Tuple, Optional, Dict
import flwr as fl
import numpy as np
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from utils import plot_metrics

class SMPCServer(fl.server.strategy.FedAvg):
    def __init__(self, num_clients: int, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0):
        super().__init__()
        self.num_clients = num_clients
        self.clients = []
        self.wait_timeout = 60 # seconds
        self.clients_ready_event = threading.Event()
        self.model_structure = None
        self.loss_per_round = []
        self.accuracy_per_round = []

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

    def aggregate_smpc_weights(self, weights_results: List[List[np.ndarray]], num_samples: List[int]) -> List[np.ndarray]:
        """Aggregate weights using Secure Multi-Party Computation (SMPC)"""
        total_samples = sum(num_samples)
        aggregated_weights = []

        print("-- SERVER SMPC AGGREGATION --")
        for layer_weight in zip(*weights_results):
            layer_shape = layer_weight[0].shape
            print(layer_shape)

            if layer_weight[0].ndim == 0:
                # Bias
                aggregated_layer = sum(w * n for w, n in zip(layer_weight, num_samples)) / total_samples
                aggregated_weights.append(aggregated_layer)
                continue

            flattened_weights = [np.array(w, dtype=np.float32).flatten() for w in layer_weight]

            # Sum up the shares
            aggregated_layers = sum(weight * num_samples[i] for i, weight in enumerate(flattened_weights))
            aggregated_layers /= total_samples  # Normalize by total samples

            aggregated_layer = aggregated_layers.reshape(layer_shape)
            aggregated_weights.append(aggregated_layer)

        return aggregated_weights

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """Aggregate the results of the clients' fit using SMPC protocol"""
        if not results:
            return None, {}

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_samples = [fit_res.num_examples for _, fit_res in results]

        # Aggregate weights using SMPC
        aggregated_weights = self.aggregate_smpc_weights(weights_results, num_samples)
        assert len(aggregated_weights) == len(weights_results[0]), "Mismatch in number of layers"

        # Aggregate metrics (if any)
        parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        metrics_aggregated = {}
        return parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the results of the clients' evaluation using SMPC protocol"""
        if not results:
            print(f"Round {server_round}: No successfull evaluations. Failures: {failures}")
            return None, {}

        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        losses = [evaluate_res.loss for _, evaluate_res in results]
        num_samples = [evaluate_res.num_examples for _, evaluate_res in results]

        # Calculate weighted average of accuracies and losses
        weighted_accuracy = np.average(accuracies, weights=num_samples)
        weighted_loss = np.average(losses, weights=num_samples)

        print(f"Round {server_round}: Weighted Accuracy: {weighted_accuracy}, Weighted Loss: {weighted_loss}")
        self.loss_per_round.append(weighted_loss)
        self.accuracy_per_round.append(weighted_accuracy)
        plot_metrics(self.loss_per_round, self.accuracy_per_round)

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
                           config=fl.server.ServerConfig(num_rounds=5))
