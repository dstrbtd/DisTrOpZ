<div align="center">

# DisTrOpZ
SN38 - Mechanism 2


<img src="docs/imagination.png" alt="EXO Gym" width="50%">

<!-- New image ideas: A macbook with a loss curve on it, with a 'thinking bubbles' coming out of the macbook, and in the bubble there is a stack of 4 H100 GPUs. It's like the laptop is imagining the cluster.  -->


##### DisTrOpZ: Distritubed Training Over The Internet Optimization

Inspired by [Exo Gym](https://github.com/exo-explore/gym), DisTrOpZ is a mechanism within Bittensor's SN38 that focuses on incentivising the creation of optimal distributed training algorithims for nodes connected via slow internet speeds.

##### How To Run A Miner

Use miner_base.py as your baseline to come up with a distributed training optimizer and communication strategy that can outperform other popular startegies. Once you have an optimize solution, publish it as a gist and commit it to the chain using the following command:


```bash
python /root/DisTrOpZ/miner/miner.py --script_path /root/DisTrOpZ/miner/miner_diloco.py --github-token <your_github_token>  --subtensor.network <test / finney / local> --netuid <your netuid> --wallet.name <your miner wallet> --wallet.hotkey <your validator hotkey>
```


##### How To Run A Validator

```bash
python /root/DisTrOpZ/validator/validator.py --subtensor.network <test / finney / local> --netuid <your netuid> --wallet.name <your miner wallet> --wallet.hotkey <your validator hotkey>
```