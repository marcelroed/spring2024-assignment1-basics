# Rust functions for CS336

## Installation
1. Ensure you have Rust on your machine. You can get it with rustup, using the system package manager, or even with conda.
`rustc` and `cargo` should be available on your machine.

2. `maturin` is used to build PyO3 code, so install that using `pip`:
```sh
pip install maturin
```

3. Build and install the Rust code:
```sh
maturin develop --release
```

I'm responsive (on Slack or otherwise), so reach out if there are any questions.