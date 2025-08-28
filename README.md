# Sketching stochastic valuation functions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sketch-EXP/Sketch/blob/main)

This repository contains datasets and numerical experiments for sketching set utility functions, introduced in our paper [Sketching stochastic valuation functions](https://arxiv.org/abs/2202.00190).

### Getting Started
We recommend one of the following options:
- **Run in Google Colab**: Click the badge above to launch an interactive Colab session with no local setup required.  
- **Run locally with Git**: Clone the repository so you can access it locally with Anaconda (git clone https://github.com/Sketch-EXP/Sketch.git). You can also click the "download" button, rather than using git. 

---

### Instructions
* [Datasets](#datasets)
  * Youtube
  * StackExchange
  * New York Times
* [Function approximation](#experiments)
  * [Synthetic data](#sd)
  * [Real-world data](#rk)
* [Best set selection](#best)

### Datasets

#### [Youtube](https://www.kaggle.com/dmitrynikolaev/youtube-dislikes-dataset)
This dataset contains information about trending YouTube videos from August 2020 to December 2021 for the USA, Canada, and Great Britain.
The clean dataset containing performance of content publishers with respect to the number of acquired views across their published content pieces is available in the dataset folder.

#### [StackExchange](https://archive.org/details/stackexchange)
This dataset contains information about 35218 questions and 88584 answers on the Academia.StackExchange platform, retrieved on Jan 20, 2022. The clean dataset, which contains questions and answers with upvote and downvote counts, is available in the dataset folder. The newest data dump can be downloaded from the official StackExchange data dump site.

#### [New York Times](https://www.kaggle.com/benjaminawd/new-york-times-articles-comments-2020)
This dataset contains all comments and articles of New York Times from January 2020 to December 2020. The clean dataset without comment details is available in the dataset folder.

### Function approximation <a name="experiments"></a>
#### Synthetic data <a name="sd"></a>

* In our experiments, we examine three utility functions (max, CES-2 and square-root of sum)
* We fix a set of 50 items and generate 10000 training samples to evaluate the set utility functions. These hyperparameters can be changed in the `framework.py` file.

**Step 1**<a name="func"></a>. Run the python scripts `framework.py` and `exp.py`. The first one contains basic code for the discretization algorithm, and the second one contains functions for running the main experiments.

**Step 2**<a name="plt"></a>. Run the notebook `plot_synthetic.ipynb`. Firstly, we illustrate how the value of epsilon effects the performance ratio. Then we compare the performance with the test score method. Finally, we show the aggregated results from all settings for different set utility functions and item value distributions.

#### Real world data <a name="rk"></a>
**Step 1**<a name="func2"></a>. Run the python scripts `framework.py` and `real.py`. These files contain similar functions as for the synthetic data case, adapted for empirical CDFs.

**Step 2**<a name="exp"></a>. Run the three notebooks with the same title as the three datasets. We show the empirical CDFs for different measures in each notebook. The performance ratios for each dataset are stored in separate files.

### Best set selection <a name="best"></a>
**Step 1**<a name="func"></a>. Run the python scripts `framework.py` and `exp.py` same as above. The files are adapted for best set selection.

**Step 2**<a name="plt"></a>. Run the notebook `selection.ipynb`. Firstly, we compared the test set selection performance with test score benchmark on various distributions. Then we discuss how the value of epsilon effects the performance ratio.
