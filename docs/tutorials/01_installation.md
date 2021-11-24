[Documentation - Home](../docs/index.md)

## 1. Installation

Developed using Python3.8. You can install the required modules by creating a
virtual environment and install the `requirements.txt`-file (run these commands
from the main folder):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

As an alternative you can install the required modules by running:

```
pip3 install numpy pandas pandas_profiling matplotlib tensorflow sklearn plotly pyyaml dvc nonconformist
```

To get a plot of the neural network architecture, the following software needs
to be installed: [Graphviz](https://graphviz.org/about/).

Next: [Quickstart](02_quickstart.md)
