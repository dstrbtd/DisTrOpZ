<div align="center">

# DisTrOpZ
SN38 - Mechanism 2


<img src="docs/imagination.png" alt="EXO Gym" width="50%">


### DisTrOpZ: Distritubed Training Over The Internet Optimization

Inspired by [Exo Gym](https://github.com/exo-explore/gym), DisTrOpZ is a mechanism within Bittensor's SN38 that focuses on incentivising the creation of optimal distributed training algorithims for nodes connected via slow internet speeds.

#### How To Run A Miner

Use miner_base.py as your baseline to come up with a distributed training optimizer and communication strategy that can outperform other popular startegies. Once you have an optimal solution, publish it as a gist and commit it to the chain using the following command:

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

<div align="center">

##### How To Run A Validator

</div>

```bash
python /root/DisTrOpZ/validator/validator.py
    --subtensor.network <test / finney / local>
    --netuid <your netuid>
    --wallet.name <your validator wallet>
    --wallet.hotkey <your validator hotkey>
```