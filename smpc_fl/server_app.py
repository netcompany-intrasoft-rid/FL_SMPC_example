"""SMPC Federated Learning Server App using Grid Messages API relay."""

import pickle
from typing import Dict, List

import numpy as np
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp


app = ServerApp()


def initialize_parameters() -> List[np.ndarray]:
    """Initialize model parameters."""
    np.random.seed(42)
    weights1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2 / 784)
    bias1 = np.zeros(128, dtype=np.float32)
    weights2 = np.random.randn(128, 10).astype(np.float32) * np.sqrt(2 / 128)
    bias2 = np.zeros(10, dtype=np.float32)
    return [weights1, bias1, weights2, bias2]


def weighted_average(weights_list: List[List[np.ndarray]], sample_counts: List[int]) -> List[np.ndarray]:
    """Compute weighted average over model weights."""
    total_samples = sum(sample_counts)
    aggregated: List[np.ndarray] = []

    for layer_values in zip(*weights_list):
        layer_sum = sum(layer * sample_counts[idx] for idx, layer in enumerate(layer_values))
        aggregated.append(layer_sum / total_samples)

    return aggregated


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run SMPC FL rounds with message relay through server grid."""
    num_rounds = int(context.run_config.get("num-server-rounds", 10))

    node_ids = sorted(list(grid.get_node_ids()))
    if not node_ids:
        raise RuntimeError("No connected nodes found. Start clients and retry.")

    num_clients = len(node_ids)
    node_to_logical = {node_id: idx for idx, node_id in enumerate(node_ids)}

    global_weights = initialize_parameters()

    for server_round in range(1, num_rounds + 1):
        # Phase 1: request local train + full share map from each client
        train_msgs: List[Message] = []
        for node_id in node_ids:
            content = RecordDict()
            content.array_records["global_params"] = ArrayRecord(global_weights)
            content.config_records["config"] = ConfigRecord(
                {
                    "logical_id": node_to_logical[node_id],
                    "num_clients": num_clients,
                    "round": server_round,
                }
            )
            train_msgs.append(
                Message(
                    content=content,
                    dst_node_id=node_id,
                    message_type="query.local_train",
                    group_id=str(server_round),
                )
            )

        train_replies = list(grid.send_and_receive(train_msgs, timeout=240.0))
        if len(train_replies) != num_clients:
            raise RuntimeError(
                f"Round {server_round}: expected {num_clients} train replies, got {len(train_replies)}"
            )

        shares_by_sender: Dict[int, Dict[int, List[np.ndarray]]] = {}
        sample_count_by_sender: Dict[int, int] = {}

        for reply in train_replies:
            sender_node = reply.metadata.src_node_id
            sender_lid = node_to_logical[sender_node]

            payload_blob = reply.content.config_records["train_reply"]["payload"]
            payload = pickle.loads(payload_blob)

            shares_by_sender[sender_lid] = payload["shares_by_recipient"]
            sample_count_by_sender[sender_lid] = int(payload["num_examples"])

        # Build per-recipient inbox of shares: each client receives one share from each sender
        shares_for_recipient: Dict[int, List[List[np.ndarray]]] = {lid: [] for lid in range(num_clients)}
        for sender_lid in range(num_clients):
            sender_map = shares_by_sender[sender_lid]
            for recipient_lid in range(num_clients):
                shares_for_recipient[recipient_lid].append(sender_map[recipient_lid])

        # Phase 2: relay shares to recipients, each client reconstructs aggregated weights
        aggregate_msgs: List[Message] = []
        for node_id in node_ids:
            recipient_lid = node_to_logical[node_id]
            payload = {
                "logical_id": recipient_lid,
                "num_clients": num_clients,
                "shares_for_me": shares_for_recipient[recipient_lid],
            }
            content = RecordDict()
            content.config_records["config"] = ConfigRecord({"payload": pickle.dumps(payload)})
            aggregate_msgs.append(
                Message(
                    content=content,
                    dst_node_id=node_id,
                    message_type="query.aggregate_shares",
                    group_id=str(server_round),
                )
            )

        aggregate_replies = list(grid.send_and_receive(aggregate_msgs, timeout=240.0))
        if len(aggregate_replies) != num_clients:
            raise RuntimeError(
                f"Round {server_round}: expected {num_clients} aggregate replies, got {len(aggregate_replies)}"
            )

        aggregated_by_client: List[List[np.ndarray]] = [None] * num_clients  # type: ignore
        sample_counts: List[int] = [0] * num_clients

        for reply in aggregate_replies:
            sender_node = reply.metadata.src_node_id
            sender_lid = node_to_logical[sender_node]

            aggregated_by_client[sender_lid] = reply.content.array_records[
                "aggregated_params"
            ].to_numpy_ndarrays()
            sample_counts[sender_lid] = sample_count_by_sender[sender_lid]

        global_weights = weighted_average(aggregated_by_client, sample_counts)

        # Phase 3: evaluate new global model
        eval_msgs: List[Message] = []
        for node_id in node_ids:
            content = RecordDict()
            content.array_records["global_params"] = ArrayRecord(global_weights)
            content.config_records["config"] = ConfigRecord(
                {
                    "logical_id": node_to_logical[node_id],
                    "num_clients": num_clients,
                    "round": server_round,
                }
            )
            eval_msgs.append(
                Message(
                    content=content,
                    dst_node_id=node_id,
                    message_type="query.evaluate_global",
                    group_id=str(server_round),
                )
            )

        eval_replies = list(grid.send_and_receive(eval_msgs, timeout=120.0))
        losses: List[float] = []
        accuracies: List[float] = []
        eval_counts: List[int] = []

        for reply in eval_replies:
            eval_cfg = reply.content.config_records["evaluate_reply"]
            losses.append(float(eval_cfg["loss"]))
            accuracies.append(float(eval_cfg["accuracy"]))
            eval_counts.append(int(eval_cfg["num_examples"]))

        weighted_loss = float(np.average(losses, weights=eval_counts))
        weighted_accuracy = float(np.average(accuracies, weights=eval_counts))
        print(
            f"Round {server_round}: Accuracy: {weighted_accuracy:.4f}, Loss: {weighted_loss:.4f}"
        )
