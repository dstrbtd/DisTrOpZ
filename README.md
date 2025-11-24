<div align="center">

# DisTrOpZ


<img src="assets/imagination.png" alt="EXO Gym" width="50%">


## DisTrOpZ: Distritubed Training Over The Internet Optimization
SN38 - Mechanism 1

</div>

Inspired by [Exo Gym](https://github.com/exo-explore/gym), DisTrOpZ is a mechanism within Bittensor's SN38 that focuses on incentivising the creation of optimal distributed training algorithims for nodes connected via slow internet speeds. Miners compete to try and produce strategies that are more efficient at dsitributed training.

Here we use Exo Gym's definition of a strategy: an abstract class for an optimization strategy, which both defines **how the nodes communicate** with each other and **how model weights are updated**. Typically, a gradient strategy will include an optimizer as well as a communication step. Sometimes (eg. DeMo), the optimizer step is comingled with the communication.

### How To Run A Miner

Use ```miner_base.py``` as your baseline to come up with a distributed training optimizer and communication strategy that can outperform the current top performing strategy. Once you have an optimal solution, publish it as a gist and commit it to the chain using the following command:

</div>

```bash
python /root/DisTrOpZ/miner/miner.py 
    --script_path /root/DisTrOpZ/miner/miner_diloco.py
    --github-token <your_github_token> 
    --subtensor.network <test / finney / local>
    --netuid <your netuid>
    --wallet.name <your miner wallet>
    --wallet.hotkey <your miner hotkey>
    --script.path <local path to your github script>
    --github.token <your github token>
```

### How To Run A Validator

```bash
python /root/DisTrOpZ/validator/validator.py
    --subtensor.network <test / finney / local>
    --netuid <your netuid>
    --wallet.name <your validator wallet>
    --wallet.hotkey <your validator hotkey>
```