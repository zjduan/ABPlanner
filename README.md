# ABPlanner
Code for our paper "An Adaptable Budget Planner for Enhancing Budget-Constrained Auto-Bidding in Online Display Advertising" in KDD 2025.

## Requirements
```text
python>=3.6
pytorch==2.3.1
argparse
tqdm
scipy
```

## Usage

First, add the project path to the PYTHONPATH environment variable.
```bash
export PYTHONPATH=$PWD 
```

To train ABPlanner in the pure simulation environment with PID auto-bidder, run the following command:
```bash
python ABPlanner/main_pure_PID.py
```

To train ABPlanner in the pure simulation environment with USCB auto-bidder, first train the USCB agent by running the following command:
```bash
python PureSimEnv/USCBAgent/main.py --save_dir PureSimEnv/USCBAgent/result/
```
Then, run the following command to train ABPlanner:
```bash
python ABPlanner/main_pure_USCB.py
```
## License

MIT