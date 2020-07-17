# Project Template
Description of the project. What does it do?

## Getting Started
Quick start tutorial on how to install, run and test the python package.

### Dependencies
Developed under `python`= 3.8.1

### Installing
Simply run
```bash
$pip install git@github.com:LeopoldGabelmann/RiskSimulator.git
```

Afterwards the package can be imported via
```python
import RiskSimulator
from RiskSimulator import <does_not_exist_yet>
```

Test the installation by running the code snippet:
```bash
$python
from RiskSimulator import <does_not_exist_yet>

test = <does_not_exist_yet>(var_1 = 'Test_Input')
test.run_mc_simulation()
```

Descripe what should happen and how one can be sure that it works.
<br /><br /><br />
For **development** purposes, instead use
```bash
pip install -e ".[dev]"
pre-commit install
```
This will install the package in editable mode, which means changes will be loaded instantly and
you don't need to re-install the package every time.


## Contributors
* Leopold Gabelmann - leopold.gabelmann@posteo.de




​
