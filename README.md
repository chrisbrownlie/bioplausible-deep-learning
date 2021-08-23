# Biologically plausible deep learning

This project implements several deep learning models using PyTorch, trains them on the CIFAR-10 image classification task, and then benchmarks them by performance and ability to predict neuron activations in macaques completing the same task - thereby giving an indication of how well they represent brain function.

This research relies heavily on the PyTorch deep learning framework for Python (Paszke et al., 2019) and the Brain Score platform (Schrimpf et al., 2018).

The project was completed in partial fulfilment of the requirements for the degree of MSc Data Science. The accompanying research paper will be made available here in the near future.

# Structure of the project
All fully-connected models used in the paper are defined in *pytorch-final-fc/*, the scripts in which are described below for clarity. Links to the relevant github repositories for the papers referenced can be found at the top of each corresponding script:
- baseline_fc.py: defines a baseline deep neural network trained using backpropagation
- dfa_fc.py: defines a DNN to be trained using direct feedback alignment - implementation adapted from Launay et al. (2019)
- fa_fc.py: defines a DNN to be trained using feedback alignment - implementation adapted from Frenkel et al. (2021)
- fun_brainscore.py: defines a function which takes in a model and returns scores for each of the public Brain Score (Schrimpf et al., 2018) benchmarks
- fun_cifar10.py: defines functions which can be used to train and test models on CIFAR-10 image classification, using SGD optimisation and CE loss
- fun_utils.py: defines a function for getting a standard transformation used in the CIFAR-10 image classification task
- models.py: creates an instance of each model before training and returning test and brain scores for each, saving necessary results to a csv file
- norse_fc.py: defines an SNN created with the *norse* Python package (Pehle & Pedersen, 2021)
- tp_fc_train.py: defines functions for training a DNN with difference target propagation - adapted from Meulemans et al. (2020)
- tp_fc.py: defines a DNN which can be trained with difference target propagation - adapted from Meulemans et al. (2020)

# Other files
*misc/* and *plots/* contain ad-hoc scripts and images which were used throughout the research.

*requirements.txt* lists all the Python packages and versions used in this project.

# Implementation notes
All models were trained on CPU only on Windows 10, except for the SNN. There were compatibility issues when installing the *norse* package so this model was trained using Windows Subsystem for Linux (Ubuntu 20.04.2) on the same machine. Note that requirements.txt was generated on Windows and so the requirement for *norse* is the last version that could be installed (0.0.4). The model trained on WSL however used norse 0.0.6 so this should be installed in order to replicate the SNN.

# References
Frenkel, C., Lefebvre, M., & Bol, D. (2021). Learning Without Feedback: Fixed Random Learning Signals Allow for Feedforward Training of Deep Neural Networks. Frontiers in Neuroscience, 15. https://doi.org/10.3389/fnins.2021.629892

Launay, J., Poli, I., & Krzakala, F. (2019). Principled Training of Neural Networks with Direct Feedback Alignment. https://arxiv.org/pdf/1906.04554.pdf

Meulemans, A., Carzaniga, F. S., Suykens, J. A. K., Sacramento, J., & Grewe, B. F. (2020). A Theoretical Framework for Target Propagation. ArXiv:2006.14331 [Cs, Stat]. https://ui.adsabs.harvard.edu/link_gateway/2020arXiv200614331M/arxiv:2006.14331

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury Google, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., Devito, Z., Raison Nabla, M., Tejani, A., Chilamkurthy, S., Ai, Q., Steiner, B., & Facebook, L. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

Pehle, C.-G., & Pedersen, J. E. (2021, January 6). Norse -  A deep learning library for spiking neural networks. Zenodo. https://zenodo.org/record/4422025#.YRPZv4hKiUk

Schrimpf, M., Kubilius, J., Hong, H., Majaj, N. J., Rajalingham, R., Issa, E. B., Kar, K., Bashivan, P., Prescott-Roy, J., Geiger, F., Schmidt, K., Yamins, D. L. K., & DiCarlo, J. J. (2018). Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like? https://doi.org/10.1101/407007