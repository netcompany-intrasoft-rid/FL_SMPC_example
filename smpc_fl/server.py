"""Legacy SMPC server entrypoint (disabled).

The supported SMPC runtime uses Flower Messages API via `smpc_fl.server_app:app`.
Run with:

    flwr run .
"""


def main() -> None:
    raise SystemExit(
        "Legacy SMPC server is disabled. "
        "Run the Messages API implementation with `flwr run .` "
        "(uses smpc_fl.server_app:app)."
    )


if __name__ == "__main__":
    main()
