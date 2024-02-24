# Overview #

This is the PyTorch code for our ICCV 2023 paper "Efficient Computation Sharing for Multi-Task Visual Scene Understanding". You can find the [paper](https://arxiv.org/pdf/2303.09663.pdf) and [Video](https://www.youtube.com/watch?v=ruMgsenxTCI&t=16s) presentation  here.

 # Introduction #

 Solving multiple visual tasks using individual models can be resource-intensive, while multi-task learning can conserve resources by sharing knowledge across different tasks.
Despite the benefits of multi-task learning, such techniques can struggle with balancing the loss for each task, leading to potential performance degradation. We present a novel computation- and parameter-sharing framework that balances efficiency and accuracy to perform multiple visual tasks utilizing individually trained single-task transformers.
Our method is motivated by transfer learning schemes to reduce computational and parameter storage costs while maintaining the desired performance. Our approach involves splitting the tasks into a base task and the other sub-tasks, and sharing a significant portion of activations and parameters/weights between the base and sub-tasks to decrease inter-task redundancies and enhance knowledge sharing. 
 ## Disclaimer ##
 
 This is research-grade code, so it's possible you will encounter some hiccups. Contact me if you encounter problems or if the documentation is unclear, and I will do my best to help.

 ## Dependencies ##

 Dependencies are managed using Conda. The environment is defined in  ``` environment.yml ```.

To create the environment, run: 

```
conda env create -f environment.yml
```

Then activate the environment with:

```
conda activate MTL_sharing
```
 

## PASCAL-Context ##

We use the same data (PASCAL-Context) as ATRC and InvPT. You can download the data by:

```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
```

Make sure you put the data in the following format:

```
data
human_parts
ImageSets
JPEGImages
normals_distill
pascal-context
sal_distill
semseg
```
## NYUD-v2 ##

Since NYUD-v2 includes data for the temporal domain as well, you can use our pre-processed and downloaded data below to simplify the process:

```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
```

# Acknowledgement #

This repository borrows partial codes from [MTI-Net](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch), [ATRC](https://github.com/brdav/atrc), [InvPT](https://github.com/prismformore/Multi-Task-Transformer/tree/3b70fcc5a4f7053a7e32a9f85da5dda670c18fba?tab=readme-ov-file), and [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE/tree/main). Many thanks to them!

# Cite #

BibTex:

```
@article{shoouri2023efficient,
  title={Efficient Computation Sharing for Multi-Task Visual Scene Understanding},
  author={Shoouri, Sara and Yang, Mingyu and Fan, Zichen and Kim, Hun-Seok},
  journal={arXiv preprint arXiv:2303.09663},
  year={2023}
}
```


