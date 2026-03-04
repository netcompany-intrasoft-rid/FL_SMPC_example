# SMPC Federated Learning

A Secure Multi-Party Computation (SMPC) protocol for federated learning using the Flower framework.

## Overview

This Flower app demonstrates federated learning with additive secret-sharing-based SMPC protocol to perform secure aggregation without exposing individual model updates.

## Installation

```bash
pip install -e .
```

## Running the App

Only the Flower Messages API implementation is supported.
Legacy scripts (`smpc_fl/client.py`, `smpc_fl/server.py`) are disabled.

### Using Flower Simulation

Run with default settings (10 rounds, 3 clients):

```bash
flwr run .
```

### Custom Configuration

Create a `pyproject.toml` configuration or pass parameters:

```bash
flwr run . --run-config num-server-rounds=15
```

### Using Flower Deployment

For production deployment:

```bash
# Start SuperLink
flower-superlink --insecure

# Start SuperNode (repeat for each client)
flower-supernode --insecure

# Run the app
flwr run . --run-config num-server-rounds=10
```

## Configuration

Key parameters in `pyproject.toml`:
- `num-server-rounds`: Number of federated learning rounds (default: 10)
- `fraction-fit`: Fraction of clients to sample for training (default: 1.0)

## How It Works

### Additive Secret Sharing in SMPC

1. Each client splits its model updates into multiple secret shares
2. Shares are returned via Messages API and relayed by the server
3. Each client reconstructs local aggregated shares from relayed messages
4. The server averages reconstructed client updates into the next global model
5. Aggregation is performed without exposing individual model updates

## Project Structure

```
smpc_fl/
├── __init__.py
├── client_app.py       # ClientApp definition
├── server_app.py       # ServerApp definition
├── smpc_client.py      # SMPC protocol implementation
├── utils.py            # Utility functions
├── peer_discovery.py   # Legacy reference only (unused)
└── smpc.proto          # Legacy reference only (unused)
```

## License

MIT License

## Funding

This project was developed as part of the [SYNTHEMA](https://synthema.eu/) project funded by the European Union's Horizon Europe Research and Innovation programme under grant agreement Nr. 101095530.
