# SemanticLIDAR

This is the CLI version of the notebooks, used for local inference or local runnning if not on Kaggle. Recommended MPS backed or Cuda GPU.

## Usage:

```
python3 main.py --help
```

Will show all the options avaliable (`train`, `validate`, `make_dataset`, `visualize`), all can be run with `--help` flag to show arguments.

Make dataset will aggerate the 3 data sources into the required dataset format for `train` and `validate` commands. Run:

```
python3 main.py make_dataset --help
```

for more information.

## Setup:

Create a virtual environment (Python 3.12 required), run: `python3 -m venv ./venv`, `source .venv/bin/activate`, `pip3 install -r requirements.txt`.