# Machine Learning Pipeline

This folder contains the scripts and data used to train the dynamics model and run the model-predictive controller.

## Contents
- `new_train.py` – train the neural network from CSV logs or an existing `.npz` dataset.
- `mpc_new.py` – real-time controller that loads the trained network and communicates with the RC car over a serial port.
- `preprocess_drift_exit.py` – utility for extracting drift-exit segments from raw logs.
- `dyn_v3.pt` / `model_dataset_v3.npz` – example trained model and the corresponding dataset.
- `light_version/` – trimmed copies of the training and MPC scripts for lightweight environments.

## Data
Raw driving logs are stored in `../full_dataset` and `../new_data`. Each CSV file contains time, position, orientation and command values. Additional features such as velocity, speed and slip angle (`beta`) are computed during preprocessing. Example plots from the dataset:

Circular test:
![circle](https://github.com/user-attachments/assets/94f1a657-74ef-4894-9f6c-31b8bb7d9d1c)
Sliding (drift with stop):
![slide](https://github.com/user-attachments/assets/c50ee339-6953-4312-8df1-19a7059232fb)
Forward (without drift):
![forward](https://github.com/user-attachments/assets/ee322a18-2517-4de8-8197-bbb8f0872032)

## Training
Adjust `CSV_GLOB` in `new_train.py` to point at the desired data folder or `.npz` file and then run:

```bash
python ML/new_train.py
```

The script saves a PyTorch checkpoint (`dyn_v3.pt`), a TorchScript model and the normalised dataset (`model_dataset_v3.npz`).

## Running the Controller
Connect the RC car to the serial port defined by `PORT` in `mpc_new.py` (default `/dev/ttyACM0`) and execute:

```bash
python ML/mpc_new.py
```

The controller switches between drift, recovery and idle modes while driving laps.

## Tests
Install dependencies from `requirements.txt` and run the unit tests with `pytest`:

```bash
pip install -r requirements.txt
pytest
```
