# Overview #

This is the PyTorch code for our ICCV 2023 paper "Efficient Computation Sharing for Multi-Task Visual Scene Understanding". You can find the [paper](https://arxiv.org/pdf/2303.09663.pdf) and [Video](https://www.youtube.com/watch?v=ruMgsenxTCI&t=16s) presentation  here.

 # Introduction #

 Solving multiple visual tasks using individual models can be resource-intensive, while multi-task learning can conserve resources by sharing knowledge across different tasks.
Despite the benefits of multi-task learning, such techniques can struggle with balancing the loss for each task, leading to potential performance degradation. We present a novel computation- and parameter-sharing framework that balances efficiency and accuracy to perform multiple visual tasks utilizing individually trained single-task transformers.
Our method is motivated by transfer learning schemes to reduce computational and parameter storage costs while maintaining the desired performance. Our approach involves splitting the tasks into a base task and the other sub-tasks, and sharing a significant portion of activations and parameters/weights between the base and sub-tasks to decrease inter-task redundancies and enhance knowledge sharing. 
 # Disclaimer #
 
 This is research-grade code, so it's possible you will encounter some hiccups. Contact me if you encounter problems or if the documentation is unclear, and I will do my best to help.

 # Dependencies #

 Dependencies are managed using Conda. The environment is defined in  ``` environment.yml ```.

To create the environment, run: 

```
conda env create -f environment.yml
```

Then activate the environment with:

conda activate MTL_sharing
 

