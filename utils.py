from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import torch
from io import BytesIO
from typing import cast
import numpy as np
from flwr.common.typing import NDArray,NDArrays, Parameters

PRIME = 2**31 - 1
SCALE_FACTOR = 1e6

def float_to_fixed_point(x: float) -> int:
    return int(x * SCALE_FACTOR) % PRIME

def fixed_to_float(x: int) -> float:
    return (x / SCALE_FACTOR) % PRIME

def partition_dataset(x, y, num_clients, client_id):
    data_size = len(x)
    partition_size = data_size // num_clients

    start = partition_size * client_id
    end = start + partition_size if client_id != num_clients - 1 else data_size

    x_client = x[start:end]
    y_client = y[start:end]
    return x_client, y_client

def load_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
    tensors = [ndarray_to_sparse_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def sparse_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    return [sparse_bytes_to_ndarray(tensor) for tensor in parameters.tensors]

def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
    bytes_io = BytesIO()

    if len(ndarray.shape) > 1:
        # convert the ndarray to sparse tensor
        ndarray = torch.tensor(ndarray).to_sparse_csr()

        # send it by utilizing the sparse matrix attributes
        # WARNING: NEVER set allow_pickle to True
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html

        np.savez(
            bytes_io, # type: ignore
            crow_indices=ndarray.crow_indices(),
            col_indices=ndarray.col_indices(),
            values=ndarray.values(),
            allow_pickle=False
        )
    else:
        np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()

def sparse_bytes_to_ndarray(tensor: bytes) -> NDArray:
    bytes_io = BytesIO(tensor)
    loader = np.load(bytes_io, allow_pickle=False)

    if "crow_indices" in loader:
        ndarray_deserialized = torch.sparse_csr_tensor(
            crow_indices=loader["crow_indices"],
            col_indices=loader["col_indices"],
            values=loader["values"]
        ).to_dense().numpy()
    else:
        ndarray_deserialized = loader
    return cast(NDArray, ndarray_deserialized)

def convert_npz_file_to_array(npz_file):
    arrays = [npz_file[key] for key in npz_file.files]
    return np.concatenate(arrays)