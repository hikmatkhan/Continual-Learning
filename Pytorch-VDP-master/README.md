# Variational Density Propagation
This repository contains a custom implementation of Variational Density Propagation written from [Dera et. Al.](https://www.researchgate.net/publication/337794189_Extended_Variational_Inference_for_Propagating_Uncertainty_in_Convolutional_Neural_Networks). This work is mostly an extension of Bayes By Backprop, but instead of estimating the variance post hoc, the variance is learned and propagated over every layer of the network, leading to a variance at each network parameter. This is achieved by defining Tensor Normal Distributions (TND) over the convolutional kernels and using the Taylor Series approximation to propagate the variance through non-linear activations. A learned variance term opens up all sorts of possiblities for machine learning namely robustness to noise, both adversarial and natural, and is perceived as uncertainty in parameters.

## Table of Contents
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Maintainers](#maintainers)
- [Acknowledgements](#Acknowledgements)
- [License](#license)

## Background
Other implementations that exist are a [Tensorflow 1.15 for MNIST](https://github.com/dimahdera/Extended-Variational-Inference-CNN) and [Tensorflow 1.15 for CIFAR10](https://github.com/dimahdera/eVI_CNN_CIFAR10), both by the original author [Extended Variational Inference for Propagating Uncertainty in Convolutional Neural Networks](https://www.researchgate.net/publication/337794189_Extended_Variational_Inference_for_Propagating_Uncertainty_in_Convolutional_Neural_Networks). With time in mind, this PyTorch implementation is processes the datasets more quickly than the original implementation.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements for
Variational Density Propagation

```bash
pip install -r ./requirements.txt
```
## Usage
For Normal Training
```python
python main.py --network CIFAR10_CONV 
```
For Continued Training
```python
python main.py --network CIFAR10_CONV --load_model $PATH --continue_train True
```
For Validation Only
```python
python main.py --network CIFAR10_CONV --load_model $PATH --continue_train False
```
Typically, larger mdels like the CIFAR10 Model need Distributed Training which can be done by utilizing PyTorch Lightning
```python
python main.py --network CIFAR10_CONV --lightning True 
```
### Required Code Parameters
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-network                   | CIFAR10_CONV  | Select Dataset Experiement                         
### Continued Training Code Parameters
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-load_model                | ''            | Path to a previously trained model checkpoint
|-continue_train            | False         | Continue to train this checkpoint model        
### General Code Parameters
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-lightning                 | True          | Use PyTorch Lightning                         
|-wandb                     | True         | Use Weights and Biases                        
|-num_workers               | 8             | Number of CPU workers to process dataset    
|-data_path                 | '../../data/' | Path to save data                         
### Training Parameters
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-epochs                    | 2             | Number of epochs                          
|-tau                       | 0.002         | KL Weight Term                            
|-clamp                     | 100000        | Loss Sigma Clamping                                  
|-batch_size                | 20            | Batch size                                
|-lr                        | 0.001         | Learning Rate                             
|-gamma                     | 0.1         | Learning Rate Scheduler Gamma             
|-lr_sched                  | [15, 30]      | Learning Rate Scheduler Milestones        
|-var_sup                   | 0.001         | Loss Variance Bias                        
|-drop_out                  | 0.2           | Network Droput                            
### Weights Initialization Parameters 
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-conv_input_mean_mu        | 0             | Mean Weight Normal Distr. Mean            
|-conv_input_mean_sigma     | 0.1           | Input Mean Weight Normal Distr. Sigma     
|-conv_input_mean_bias      | 0.000001      | Input Mean Bias                           
|-conv_input_sigma_min      | -5            | Input Sigma Weight Uniform Distr. Min     
|-conv_input_sigma_max      | -2.2          | Input Sigma Weight Uniform Distr. Max     
|-conv_input_sigma_bias     | 0.000001      | Input Sigma Bias                          
|-fc_input_mean_mu          | 0             | Mean Weight Normal Distr. Mean            
|-fc_input_mean_sigma       | 0.1           | Input Mean Weight Normal Distr. Sigma     
|-fc_input_mean_bias        | 0.000001      | Input Mean Bias                           
|-fc_input_sigma_min        | -5            | Input Sigma Weight Uniform Distr. Min     
|-fc_input_sigma_max        | -2.2          | Input Sigma Weight Uniform Distr. Max     
|-fc_input_sigma_bias       | 0.000001      | Input Sigma Bias                          
### Weights and Biases Parameters
| Parameter                 | Default       | Description   |	
| :------------------------ |:------------- | :-----------------------------------------|
|-project                   | None          | WandB Project                              
|-account                   | None          | WandB Account                          
 
## Results

**Coming soon**

## Maintainers

[@angelinic0](https://github.com/angelinic0).

## Acknowledgements

Thank you to [Rowan University](https://www.rowan.edu/) and the [United States Department of Education](https://www.ed.gov/) for hosting me for my PhD Research and for funding my education through the [GAANN Fellowship](https://www2.ed.gov/programs/gaann/index.html), respectively. 


## License
Copyright 2021 Christopher Francis Angelini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
