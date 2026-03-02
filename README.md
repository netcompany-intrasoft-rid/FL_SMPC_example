## P2P SMPC Protocol for federated learning
This repository demonstrates the application of a Peer-to-Peer Secure Multi-Party Computation (P2P SMPC) protocol for federated learning, leveraging the Flower framework and gRPC for P2P communication.

### Overview
Federated learning enables decentralized training of machine learning models without sharing raw data. However, traditional federated learning still requires a central server to aggregate model updates. This project introduces an additive secret-sharing-based P2P SMPC protocol to perform secure aggregation without relying solely on a central aggregator.

### Project structure
```
.
├── README.md
├── normal_fl
│   ├── client.py
│   ├── metrics.jpg
│   ├── server.py
│   ├── start_clients_10.sh
│   ├── start_clients_3.sh
│   ├── start_clients_4.sh
│   ├── start_clients_5.sh
│   ├── start_clients_6.sh
│   ├── start_clients_7.sh
│   ├── start_clients_8.sh
│   ├── start_clients_9.sh
│   └── utils.py
├── requirements.txt
└── smpc_fl
    ├── client.py
    ├── metrics.jpg
    ├── peer_discovery.py
    ├── server.py
    ├── smpc.proto
    ├── smpc_pb2.py
    ├── smpc_pb2_grpc.py
    ├── start_clients_10.sh
    ├── start_clients_3.sh
    ├── start_clients_4.sh
    ├── start_clients_5.sh
    ├── start_clients_6.sh
    ├── start_clients_7.sh
    ├── start_clients_8.sh
    ├── start_clients_9.sh
    └── utils.py
```

### Project setup

1. Clone the repository:
   ```sh
   git clone https://github.com/netcompany-intrasoft-rid/FL_SMPC_example
   cd FL_SMPC_example
   ```
2. Install dependencies (ideally in a fresh Python environment):
   ```sh
   pip install -e .
   ```
   Or install from requirements.txt:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Generate gRPC files if modifications are made to `smpc.proto`:
   ```sh
   cd smpc_fl
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. smpc.proto
   ```

## Running the Project

### Option 1: Using Flower Hub (Recommended)

The project is Flower Hub compatible with P2P SMPC support:

**Simulation Mode** (easiest for testing):
```sh
# Run with default settings (10 rounds, 3 clients)
flwr run .

# Run with custom configuration
flwr run . --run-config num-server-rounds=5
```

**Deployment Mode** (for production):
```sh
# Terminal 1: Start SuperLink (server)
flower-superlink --insecure

# Terminal 2, 3, 4...: Start SuperNodes (one per client)
flower-supernode --insecure

# Terminal N: Run the app
flwr run . --run-config num-server-rounds=10
```

**Features:**
- ✅ P2P SMPC protocol with gRPC secret sharing
- ✅ Automatic peer discovery and connection
- ✅ Works in both simulation and deployment modes
- ✅ Uses `flwr-datasets` for automatic data partitioning

**Configuration Options:**

Edit `pyproject.toml` to customize:
```toml
[tool.flwr.app.config]
num-server-rounds = 10        # Number of training rounds
fraction-fit = 1.0            # Fraction of clients per round
base-port = 50051             # Base port for P2P communication

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3    # Number of clients in simulation
```

### Option 2: Legacy Mode

Be sure to specify the number of clients (3-10) in server and clients scripts in both implementations (normal FL and SMPC FL). The number of federated learning rounds can also be specified using the `--num_rounds` argument in the `server.py` script for both implementations (default value is 10).

#### Normal Federated Learning

1. Start the central FL server:
   ```sh
   python normal_fl/server.py --num_clients={number of clients}
   ```
2. Start multiple clients:
   ```sh
   bash normal_fl/start_clients_{number of clients}.sh
   ```

#### P2P SMPC Federated Learning

1. Start the SMPC FL server:
   ```sh
   python smpc_fl/server.py --num_clients={number of clients}
   ```
2. Start multiple clients:
   ```sh
   bash smpc_fl/start_clients_{number of clients}.sh
   ```

## How It Works

### Additive Secret Sharing in SMPC

1. Each client **splits its model updates** into multiple secret shares.
2. These shares are **distributed to different peers** in the network.
3. Each peer aggregates locally the received shares
3. The server **aggregates the locally aggregated parameters** to reconstruct the final model update.
4. The aggregation is performed **without exposing individual model updates**.

### Peer Discovery Mechanism
Clients use `peer_discovery.py` to automatically identify and connect with available peers, eliminating the need for a predefined network topology. This feature is optional. If not used, the IP addresses and ports of peers must be manually specified using the `--peer_addresses` argument in a comma-separated string format when running the `client.py` script in the SMPC implementation.

## Performance Metrics
The `metrics.jpg` files in `normal_fl/` and `smpc_fl/` visualize the performance (accuracy and loss) between **normal FL** and **P2P SMPC FL** implementations.

## License
This project is open-source under the **MIT License**.

## Funding
This project was developed as part of the [SYNTHEMA](https://synthema.eu/) project funded by the European Union’s Horizon Europe Research and Innovation programme under grant agreement Nr. 101095530. 