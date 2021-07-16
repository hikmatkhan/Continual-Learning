# Continual-Learning
## Github Commands
- [Commands](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

## Outlines
- Classics 
- Empirical Study 
- Surveys 
- Influentials 
- New Settings or Metrics 
- Regularization Methods 
- Distillation Methods 
- Rehearsal Methods 
- Generative Replay 
- Methods Dynamic Architectures or Routing Methods
- Hybrid Methods
- Continual Few-Shot Learning 
- Meta Learning 
- Meta-Continual Learning 
- Lifelong Reinforcement Learning 
- Continual Generative Modeling 
- Applications 
- Thesis 
- Libraries 
- Workshops
- Benchmarks

</br>

## Classics 
- [Catastrophic forgetting in connectionist networks](https://www.sciencedirect.com/science/article/pii/S1364661399012942), (1999) by French, Robert M.
- [Lifelong robot learning](https://www.sciencedirect.com/science/article/pii/092188909500004Y), (1995) Sebastian ThrunaTom, M.Mitchellb.
## Empirical Study 
## Surveys 
## Influentials 
## New Settings or Metrics 
## Regularization Methods 
## Distillation Methods 
## Rehearsal Methods 
## Generative Replay 
## Methods Dynamic Architectures or Routing Methods
## Hybrid Methods
## Continual Few-Shot Learning 
## Meta Learning 
- [Rapid learning or feature reuse? towards understanding the effectiveness of maml](https://arxiv.org/pdf/1909.09157.pdf) ICLR (2020) Raghu, A., Raghu, M., Bengio, S., & Vinyals.
## Meta-Continual Learning 
## Lifelong Reinforcement Learning 
## Continual Generative Modeling 
## Applications 
## Thesis 
## Libraries 
## Workshops
## Benchmarks
### [A Procedural World Generation Framework for Systematic Evaluation of Continual Learning](https://arxiv.org/abs/2106.02585), (2021) Hess, Timm, Martin Mundt, Iuliia Pliushch, and Visvanathan Ramesh.
- > Several families of continual learning techniques have been proposed to alleviate catastrophic interference in deep neural network training on non-stationary data. However, a comprehensive comparison and analysis of limitations remains largely open due to the inaccessibility to suitable datasets. Empirical examination not only varies immensely between individual works, it further currently relies on contrived composition of benchmarks through subdivision and concatenation of various prevalent static vision datasets. **In this work, our goal is to bridge this gap by introducing a computer graphics simulation framework that repeatedly renders only upcoming urban scene fragments in an endless real-time procedural world generation process.** At its core lies a modular parametric generative model with adaptable generative factors. The latter can be used to flexibly compose data streams, which significantly facilitates a detailed analysis and allows for effortless investigation of various continual learning schemes.
- >we conjecture that: if catastrophic forgetting cannot be circumvented in scenarios with a known synthetic data foundation, there is limited hope to understand limitations and overcome the challenge in real-world settings.
- ![Benchmark](/Images/b1.png?raw=true)
- In principle, the idea to leverage virtual data has already found countless prior applications, primarily due to simulators’ ability to yield automatic precise ground truth information.
- Computer-graphics frameworks thus typically facilitate sampling of a maximum of conceivable variations to enable deep learning in domains where scares data is available and real-world data acquisition is insurmountable. However, in continual learning, scenarios of interest should inherently encompass knowledge about the detailed temporal shifts in the observed distribution.These range from occurrence of particular objects, their geometry and texture, the frequency and order of objects’ (dis-)appearance in the scene, or continuous changes in the environmental weather and lighting. Corresponding investigations of continual learning thus require straightforward accessibility to meticulous control of the real-time online changes in the independent generative factors.
- our primary contribution is the introduction of a modular Unreal Engine 4 based 3-D computer graphics simulator that now also enables clear-cut generation and assessment of diverse continual learning scenarios.
- Contributions
- Introduce a simulator that facilitates grounded investigation of continual learning mechanisms through access to highly customizable data. At all times, our simulator only renders an upcoming segment of the world through efficient real-time scene assembly. Its offered data generation is based on manipulation of temporal priors and parameters of the generative model. Our modular control spans aspects from physics-based (de-)activation of color, surface normals and scattering, to switches in weather conditions or environment lighting, and ultimately to commonly evaluated abrupt changes in the data population though (dis-)appearance of entire object categories.
- Corroborate our simulator’s utility in an initial showcase of multiple continual learning techniques, investigated across incremental class, environmental lighting, and weather scenarios.
- Provide open access to: 1. As a benchmark creation tool, a stand-alone simulator executable with configuration files for the specification of rendered sequences: https://doi.org/10.5281/zenodo.4899294 2. To allow extensions, the underlying source-code of the simulator: https://github.com/ccc-frankfurt/EndlessCL-Simulator-Source 3. A set of respective videos and their precise dataset versions to reproduce the particular experiments of this paper: https://doi.org/10.5281/zenodo.4899267.
- we now generate and investigate video sequences in three distinct set-ups: incremental class appearance, varying weather conditions and decreasing illumination intensity.
- ![Benchmark](/Images/b2.png?raw=true)
- ![Benchmark](/Images/b3.png?raw=true)
- ![Benchmark](/Images/b4.png?raw=true)

