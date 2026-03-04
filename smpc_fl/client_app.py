"""SMPC Federated Learning Client App using Flower Messages API handlers."""

import logging
import pickle
from typing import Any

import numpy as np
from flwr.client import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr_datasets import FederatedDataset

from smpc_fl.smpc_client import SMPCProtocol
from smpc_fl.utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = ClientApp()


def load_data_simulation(partition_id: int, num_partitions: int):
    """Load data for simulation mode using flwr_datasets."""
    fds = FederatedDataset(dataset="mnist", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42 + partition_id)

    x_train = np.array(partition_train_test["train"]["image"]).astype("float32") / 255.0
    y_train = np.array(partition_train_test["train"]["label"])
    x_test = np.array(partition_train_test["test"]["image"]).astype("float32") / 255.0
    y_test = np.array(partition_train_test["test"]["label"])

    return x_train, y_train, x_test, y_test


def load_data_deployment():
    """Load data for deployment mode from local MNIST."""
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def _load_client_data(context: Context, num_clients: int):
    """Load client-local dataset split."""
    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
        partition_id = int(context.node_config["partition-id"])
        partition_count = int(context.node_config["num-partitions"])
        return load_data_simulation(partition_id, partition_count)

    return load_data_deployment()


@app.query("local_train")
def query_local_train(message: Message, context: Context) -> Message:
    """Train locally and return all SMPC shares for relay by the server."""
    cfg = message.content.config_records["config"]
    logical_id = int(cfg["logical_id"])
    num_clients = int(cfg["num_clients"])
    round_id = int(cfg["round"])

    global_weights = message.content.array_records["global_params"].to_numpy_ndarrays()

    x_train, y_train, x_test, y_test = _load_client_data(context, num_clients)

    model = load_model()
    model.set_weights(global_weights)
    model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)
    local_weights = model.get_weights()

    smpc = SMPCProtocol(num_clients)
    shares_by_recipient = smpc.split_weights_to_shares(local_weights, num_clients)

    payload: dict[str, Any] = {
        "logical_id": logical_id,
        "num_examples": int(len(x_train)),
        "shares_by_recipient": shares_by_recipient,
    }

    reply = RecordDict()
    reply.config_records["train_reply"] = ConfigRecord({"payload": pickle.dumps(payload)})

    logger.info(
        "Client logical_id=%s completed local_train round=%s with %s samples",
        logical_id,
        round_id,
        len(x_train),
    )
    return Message(reply, reply_to=message)


@app.query("aggregate_shares")
def query_aggregate_shares(message: Message, context: Context) -> Message:
    """Aggregate relayed shares locally and return aggregated model weights."""
    cfg = message.content.config_records["config"]
    payload = pickle.loads(cfg["payload"])

    logical_id = int(payload["logical_id"])
    num_clients = int(payload["num_clients"])
    shares_for_me = payload["shares_for_me"]

    smpc = SMPCProtocol(num_clients)
    aggregated_weights = smpc.reconstruct_weights(shares_for_me)

    reply = RecordDict()
    reply.array_records["aggregated_params"] = ArrayRecord(aggregated_weights)
    reply.config_records["aggregate_meta"] = ConfigRecord({"logical_id": logical_id})

    logger.info("Client logical_id=%s reconstructed aggregated weights", logical_id)
    return Message(reply, reply_to=message)


@app.query("evaluate_global")
def query_evaluate_global(message: Message, context: Context) -> Message:
    """Evaluate current global model and return metrics."""
    cfg = message.content.config_records["config"]
    logical_id = int(cfg["logical_id"])

    global_weights = message.content.array_records["global_params"].to_numpy_ndarrays()

    x_train, y_train, x_test, y_test = _load_client_data(context, int(cfg["num_clients"]))

    model = load_model()
    model.set_weights(global_weights)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    reply = RecordDict()
    reply.config_records["evaluate_reply"] = ConfigRecord(
        {
            "logical_id": logical_id,
            "num_examples": int(len(x_test)),
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
    )

    logger.info(
        "Client logical_id=%s evaluation: loss=%.4f acc=%.4f",
        logical_id,
        float(loss),
        float(accuracy),
    )
    return Message(reply, reply_to=message)
