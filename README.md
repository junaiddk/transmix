# TransMix
This code repository implements the method presented in "*TransMix: Transformer-based Value Function Decomposition for Cooperative Multi-agent Reinforcement Learning*". This repo is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced. 
The TransMix specific implementation is in the *src/modules/mixers*. Other algorithms available in this repo are:

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py --config=tmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=tmix --env-config=sc2 with env_args.map_name=2s3z
```

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

### TranxMix specific parameters

TransMix specific parameters can be found in *src/config/algs -> tmix.yml*

We use the parallel runner environment from PyMARL to speedup the training. Other parameters are:
- `is_noise`: to make the the global states noisy. 
- `embed_dim`: is the dim used for transformation in the transformer
- `ff`: is the feed forward dim for the transformer
- `t_depth`: represents the number of transformer encoder layers
- `heads`: shows the number of multi-heads inside the transformer encoder

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

# Replays
Here are some example replays/results of the trained agents on one of the hard scenarios of SMAC:

> 8m_vs_9m 

https://user-images.githubusercontent.com/45826429/195468265-58678a3b-4945-4302-8729-fdca06227318.mp4

> MMM

https://user-images.githubusercontent.com/45826429/196526347-d368488b-ae18-4cc3-94f0-6241c1192992.mp4

> 2c_vs_64zg

https://user-images.githubusercontent.com/45826429/196527347-1dbbf7c1-58dc-48ff-9299-b3b0667fdf2e.mp4



# Citing TranxMix

Please cite the paper as (in BibTex format):
```
@article{Khan_Ahmed_Sukthankar_2022, \
     title={Transformer-Based Value Function Decomposition for Cooperative Multi-Agent Reinforcement Learning in StarCraft}, \
     volume={18}, \
     url={https://ojs.aaai.org/index.php/AIIDE/article/view/21954}, \
     journal={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment}, author={Khan, Muhammad Junaid and Ahmed, Syed Hammad      and Sukthankar, Gita}, \
     year={2022}, \
     pages={113-119} \
}
```
