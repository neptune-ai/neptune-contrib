# Python lib
[![Build Status](https://travis-ci.org/neptune-ml/neptune-lib.svg?branch=master)](https://travis-ci.org/neptune-ml/neptune-lib)

# Getting started

## Experiment Api

### Get api-token

```bash
neptune account api-token get
```

### Establish session

```python
from neptunelib.session import Session
session = Session(api_token='YOUR_NEPTUNE_API_TOKEN')
```

you can also create an environment variable `NEPTUNE_API_TOKEN`:

```bash
export NEPTUNE_API_TOKEN=YOUR_NEPTUNE_API_TOKEN`
```

and simpy go

```python
from neptunelib.session import Session
session = Session()
```

### Instatiate a Project object

```python
project = session.get_projects('neptune-ml')['neptune-ml/Salt-Detection']
```

### Get leaderbaord dataframe

```python
leaderboard_df = project.get_leaderboard()
```

### Get experiment data

```python
experiments = project.get_experiments(id=['SAL-2342'])
experiment = experiments[0]
```

1. get numeric channel values dataframe
    
```python
    channel_df = experiment.get_numeric_channels_values('network_1 epoch_val iou loss')
```
    
2. get hardware utilization dataframe 

```python
   channel_df = experiment.get_hardware_utilization()
```


# Installation

## Get prerequisites
* python versions `2.7/3.5/3.6` are supported
* `neptune-cli` 
   ```python
      pip install neptune-cli
   ```

## Install lib

```bash
pip install neptune-lib
```

# Getting help
If you get stuck, don't worry we are here to help.
The best order of communication is:

 * [neptune community forum](https://community.neptune.ml/)
 * Github issues
 
# Contributing
If you see something that you don't like you are more than welcome to contribute!
There are many options:
  
  * Participate in discussions on [neptune community forum](https://community.neptune.ml/)
  * Submit a feature request or a bug here, on Github
  * Submit a pull request that deals with an open feature request or bug
  * Spread a word about neptune-lib in your community
