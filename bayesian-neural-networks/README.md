# Bayesian Neural Networks — Neural Networks with Caution

A hands-on course in Jupyter notebooks that accompanies the text *Bayesian Neural Networks: Neural Networks with Caution* (Hale Türeli, Mathematics, 2025).
Each notebook follows one chapter, mixes explanation and equations with runnable Python, and ends with exercises.

**Website:** `https://USERNAME.github.io/bayesian-neural-networks/` (after the one-time setup below)

## The course

| # | Notebook | Chapter | Tools | Open |
|---|---|---|---|---|
| 00 | [Why uncertainty matters](notebooks/00_why_uncertainty_matters.ipynb) | Introduction | scikit-learn | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/00_why_uncertainty_matters.ipynb) |
| 01 | [Probability and Bayes' rule](notebooks/01_probability_and_bayes_rule.ipynb) | Probability | NumPy, SciPy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/01_probability_and_bayes_rule.ipynb) |
| 02 | [The mathematics of neural networks](notebooks/02_mathematics_of_neural_networks.ipynb) | Mathematics of NNs | NumPy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/02_mathematics_of_neural_networks.ipynb) |
| 03 | [MLE vs MAP estimation](notebooks/03_mle_vs_map_estimation.ipynb) | Bayesian NNs: optimisation, MLE, MAP | NumPy, SciPy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/03_mle_vs_map_estimation.ipynb) |
| 04 | [Weights as probability distributions](notebooks/04_weights_as_distributions.ipynb) | Bayesian NNs: weights as distributions | NumPy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/04_weights_as_distributions.ipynb) |
| 05 | [A BNN via MCMC](notebooks/05_bnn_mcmc_numpy.ipynb) | Approximations: sampling | NumPy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/05_bnn_mcmc_numpy.ipynb) |
| 06 | [Bayes by Backprop](notebooks/06_bnn_bayes_by_backprop_pytorch.ipynb) | Approximations: variational inference | PyTorch | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/06_bnn_bayes_by_backprop_pytorch.ipynb) |
| 07 | [MC dropout and deep ensembles](notebooks/07_mc_dropout_and_ensembles_pytorch.ipynb) | Approximations: practical methods | PyTorch | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/07_mc_dropout_and_ensembles_pytorch.ipynb) |
| 08 | [BNNs with TensorFlow Probability](notebooks/08_bnn_tensorflow_probability.ipynb) | Software for BNN: TensorFlow | TensorFlow Probability | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/08_bnn_tensorflow_probability.ipynb) |
| 09 | [Safety-critical predictions and decisions](notebooks/09_applications_safety_and_decisions.ipynb) | Applications of BNN | PyTorch, scikit-learn | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/bayesian-neural-networks/blob/main/notebooks/09_applications_safety_and_decisions.ipynb) |

Notebooks 00–05 need only the scientific Python stack. Every notebook runs on a laptop CPU in under about two minutes and is committed with its outputs, so you can read it on GitHub without running anything.

## Run locally

```bash
git clone https://github.com/USERNAME/bayesian-neural-networks.git
cd bayesian-neural-networks
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Publish the website (one-time setup)

1. Create a public repository named `bayesian-neural-networks` on GitHub and push these files to the `main` branch.
2. In the repository, open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. The workflow in `.github/workflows/pages.yml` runs on every push. It renders the notebooks to HTML and publishes them with `index.html`.
4. Replace `USERNAME` in this README with your GitHub user name. The website detects the repository address by itself.

## Other notebooks worth studying

- Chandra & Simmons, *Bayesian Neural Networks via MCMC: A Python-Based Tutorial* — companion code: <https://github.com/sydney-machine-learning/Bayesianneuralnetworks-MCMC-tutorial>
- UvA Deep Learning tutorials — *Bayesian Deep Learning* and the DL2 *Bayesian Neural Networks with Pyro* series: <https://uvadlc-notebooks.readthedocs.io>
- Pyro and PyTorch BNN on MNIST ("Getting your neural network to say *I don't know*"): <https://github.com/paraschopra/bayesian-neural-network-mnist>
- Kevin Murphy's *Probabilistic Machine Learning* book notebooks: <https://github.com/probml/pyprobml>

## References

- Baan, J. (2021). *A Comprehensive Introduction to Bayesian Deep Learning*.
- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. *ICML*.
- Brownlee, J. (2020). *Probability for Machine Learning*.
- Chandra, R., & Simmons, J. (2024). Bayesian Neural Networks via MCMC: A Python-Based Tutorial. *IEEE Access, 12*, 70519–70549.
- Costa, G. (2022). *A first insight into Bayesian Neural Networks (BNNs)*.
- Diaz Ochoa, J. G., Maier, L., & Csiszar, O. (2023). Bayesian logical neural networks for human-centered applications in medicine. *Frontiers in Bioinformatics, 3*.
- Foong, A. Y. K., Li, Y., Hernández-Lobato, J. M., & Turner, R. E. (2019). "In-between" uncertainty in Bayesian neural networks. *ICML Workshop on Uncertainty and Robustness in Deep Learning*.
- Gal, Y. (2016). *Uncertainty in Deep Learning*. PhD thesis, University of Cambridge.
- Hein, M., Andriushchenko, M., & Bitterwolf, J. (2019). Why ReLU networks yield high-confidence predictions far away from the training data. *CVPR*.
- Jospin, L. V., Laga, H., Boussaid, F., Buntine, W., & Bennamoun, M. (2022). Hands-On Bayesian Neural Networks — A Tutorial for Deep Learning Users. *IEEE Computational Intelligence Magazine, 17*(2), 29–48.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.

## License

Code is released under the MIT License (see `LICENSE`).
