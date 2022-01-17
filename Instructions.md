# Instructions
* [Datasets](#datasets)
  * [Youtube](#youtube)
  * [StackExchange](#stackex)
  * [New York Times](#times)
* [Numerical Experiments](#experiments)
  * [Synthetic data](#sd)
    * [Main functions](#func)
    * [Plots](#plt)
  * [Real-world data](#rk)
    * [Main functions](#func2)
    * [Main experiments](#exp)

## Datasets

### [Youtube](https://www.kaggle.com/dmitrynikolaev/youtube-dislikes-dataset)
This dataset contains information about trending YouTube videos from August 2020 to December 2021 for the USA, Canada, and Great Britain.
The clean dataset containing performance of content publishers with respect to the number of acquired views across their published content pieces is available in the dataset folder.
### [StackExchange](https://archive.org/details/stackexchange)
The clean dataset, which contains questions and answers with upvote and downvote counts, is available in the dataset folder. The newest data dump can be downloaded from the official StackExchange data dump site.

### [New York Times](https://www.kaggle.com/benjaminawd/new-york-times-articles-comments-2020)
This dataset contains all comments and articles of New York Times from January 2020 to December 2020. The clean dataset without comment details is available in the dataset folder.

## Numerical Experiments <a name="experiments"></a>
### Synthetic data <a name="sd"></a>

* In our experiments, we examine three utility functions (max, CES-2 and square-root of sum)
* We fix a set of 50 items and generate 500 training samples to evaluate the set utility functions. These hyperparameters can be changed in the framework.py file.

#### Step-by-step instructions:

**Step 1**<a name="func"></a>. Run the python scripts framework.py and exp.py. The first one contains basic code for the discretization algorithm, and the second one contains functions for running the main experiments.

**Step 2**<a name="plt">. Run the notebook plot.ipynb. Firstly, we illustrate how the value of epsilon effects the performance ratio. Then we compare the performance with the test score method. Finally, we show the aggregated results from all settings for different set utility functions and item value distributions.


### Real world data <a name="rk"></a>

#### Step-by-step instructions:

**Step 1**<a name="func2"></a>. Run the python scripts framework.py and real.py. These files contains similar functions as for the synthetic data case, adapt for empirical CDFs.

**Step 2**<a name="exp"></a>. Run the three notebooks for the three datasets. We show the emprical CDFs for different measures in each notebook. The performance ratios for each dataset are stored in saparate files. 
