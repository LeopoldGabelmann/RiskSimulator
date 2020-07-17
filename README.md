# RiskSimulator
As a kid I often played the bord game risk with my friends and back then we were always discussing over our mutual feeling that the game is set up so that the attacker has better odds than the defender. To varify our guess, we startet to design an entire table with all possible throw combinations to determine the precise probabilities of losing and winning for the attacker and the defender. As you can imagine, we never finished our table since there are 6^5 (7776) possible combinations. <br>
Having played risk recently and basically rampaging all my friends by simply always being the aggressor, I remembered our childhood struggle to varify the true probabilites of winning, if being the attacker or the defender in risk. But this time solving the problem with brute force calculation, the power of bootstrapping and the central limit theorem.

## Getting Started
Quick start tutorial on how to install, run and test the python package.

### Dependencies
Developed under `python`= 3.8.1

### Installing
Simply run
```bash
$pip install git@github.com:LeopoldGabelmann/RiskSimulator.git
```

To test the installation you can run the following code snippet to run one bootstrap:
```python
import RiskSimulator
from RiskSimulator import run_one_bootstrap

simulator = MonteCarloSimulator(defender_range(0,6), attacker_range(0,6))
count_attacker, count_defender = simulator.run_one_bootstrap(nruns=10, verbose=1)
```
This code is running one bootstrap and returning the results in absolute numbers.

If you want to run the entire simulation you can adjust the above code to:
```python
import RiskSimulator
from RiskSimulator import run_bootstrap_monte_carlo

simulator = MonteCarloSimulator(defender_range(0,6), attacker_range(0,6))
mean_attacker, mean_defender = simulator.run_bootstrap_monte_carlo(n_bootstraps = 100, nruns=10, verbose=10)
```
This way you can run the entire simulation over multiple bootstraps. The method is returning the average result over all bootstraps.

<br /><br />
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
