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
import project_template
from project_template import ClassTemplate
```

Test the installation by running the code snippet:
```bash
$python
from project_template import ClassTemplate

test = ClassTemplate(var_1 = 'Test_Input')
test.example_method()
```

Descripe what should happen and how one can be sure that it works.

For development purposes, instead use
```bash
pip install -e ".[dev]"
pre-commit install
```
This will install the package in editable mode, which means changes will be loaded instantly and
you don't need to re-install the package every time.

### Running the classes
`ClassTemplate()`:

Class Description: What does it do?

| Variable      | Type   | Input                                                        |
| ------------- | ------ | ------------------------------------------------------------ |
| `var_1`        | string | How to define the variable and how the class uses it.       |
| `loglevel`  | string | Define the log level to determine which logs are printed.  Has to be one of `NOTSET`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `INFO`. |


## Running tests
Tests are run with the pytest runner. In total ? unit tests are conducted. To run the tests enter

```bash
pytest
```

### Unit Tests
The class `TestClass` tests what? Description.

### Integration Tests
If applicable - Description.

### Test coverage
To show the test coverage run from the command line
```bash
 pytest --cov=project_template
```


## Code Style corrections
To auto-correct all python code run from the command line in the root directory
``` bash
pip install autopep8==1.4.4
autopep8 --in-place --aggressive --aggressive --recursive . --max-line-length 100 -v
```

To rate the code run form the command line in the root directory
```bash
pip install pylint==2.4.2
pylint -j 3 project_template\project_template.py tests\test_unittest.py --disable=no-self-use
```

To check versus the project desired threshold of 9.0, run from the command line:
```bash
pip install pylint==2.4.2
pip install pylint-fail-under==0.3.0
pylint-fail-under --fail_under 9.0 -j 3 project_template\project_template.py tests
\test_unittest.py --disable=no-self-use
```


## Contributors
* Leopold Gabelmann - leopold.gabelmann@posteo.de




​
