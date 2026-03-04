## P2P SMPC Protocol for federated learning
This repository demonstrates Secure Multi-Party Computation (SMPC) for federated learning using Flower (v1.26.1) and Flower's Messages API.

### Overview
Federated learning enables decentralized training of machine learning models without sharing raw data. However, traditional federated learning still requires a central server to aggregate model updates. This project introduces an additive secret-sharing-based P2P SMPC protocol to perform secure aggregation without relying solely on a central aggregator.

### Key Features
- ✅ **Messages API SMPC**: Share exchange is orchestrated via ServerApp message relay
- ✅ **Additive Secret Sharing**: Secure multi-party computation for privacy-preserving aggregation
- ✅ **Simulation & Deployment**: Works in both modes seamlessly

### Project structure
```
.
├── README.md
├── requirements.txt
└── smpc_fl
    ├── client_app.py      # Client with Flower Message APIs
    ├── server_app.py      # Server strategy
    ├── smpc_client.py     # SMPC protocol implementation
    └── utils.py           # Helper functions
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

## Running the Project

### Option 1: Using Flower Hub (Recommended)

The project is Flower Hub compatible with SMPC support using Flower's native Message APIs:

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
- ✅ Messages API SMPC protocol with server-side message relay
- ✅ Clients locally reconstruct aggregated shares
- ✅ Works in both simulation and deployment modes

**Configuration Options:**

Edit `pyproject.toml` to customize:
```toml
[tool.flwr.app.config]
num-server-rounds = 10        # Number of training rounds
fraction-fit = 1.0            # Fraction of clients per round

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3    # Number of clients in simulation
```

## How It Works

### Additive Secret Sharing in SMPC

1. Each client **trains locally** and obtains model updates.
2. Each client **splits its model updates** into N secret shares (where N = number of clients).
3. Each client returns per-recipient shares through **Flower Messages API**.
4. The server relays shares to each recipient client via **Flower Messages API**.
5. Each client **sums all received shares** (including its own) to get the aggregated model.
6. Clients send their **locally aggregated results** to the server.
7. The server performs **weighted averaging** of the aggregated results.

### SMPC with Flower Message API
In Flower 1.26.1, we implement SMPC using the Message API relay pattern:
- Clients split their weights into shares using additive secret sharing
- ServerApp sends/receives `query.*` messages via Grid and relays shares
- Each client locally aggregates all shares (own + received)
- The server receives already-aggregated weights and performs weighted averaging
- This provides privacy as no single entity sees individual client updates

## License
This project is open-source under the **MIT License**.

## Funding
This project was developed as part of the [SYNTHEMA](https://synthema.eu/) project funded by the European Union’s Horizon Europe Research and Innovation programme under grant agreement Nr. 101095530. 