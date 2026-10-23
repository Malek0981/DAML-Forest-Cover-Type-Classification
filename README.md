# DAML-Forest-Cover-Type-Classification

Based on the Dataset : https://archive.ics.uci.edu/dataset/31/covertype
(Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N.)

The goal here is to predict forest cover type from cartographic variables only. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.  These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

We will :
- Explore the data to look for null/incoherent values
- Build a simple model to have first results
- Build more complex models
- Explore the data again to see if we can get rid of some noisy features
- Adapt our models
- Benchmark them
- Analyze the results

# To use this code :

## Download uv : 

#### On macOS and Linux
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```


#### On Windows
```bash 
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### With pip
```bash
pip install uv
```

#### With Homebrew.
```bash
brew install uv
```


## Install dependencies
In your terminal, run  :
```bash
uv sync
```