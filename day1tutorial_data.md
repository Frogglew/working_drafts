# Access and explore data in Azure Machine Learning

In this tutorial you'll learn how to:

> * Different ways to prepare your data on Azure Machine Learning
> * How to load data from an existing cloud storage
> * How to preview the data
> * Access data in a notebook for interactive development
> * Process data
> * Create new versions of data assets

## Prerequisites

* Complete the [Quickstart: Get started with Azure Machine Learning](quickstart-create-resources.md) to:
    * Create a workspace.
    * Create a cloud-based compute instance to use for your development environment.

## Bring data to use with Azure Machine Learning
In most cases, you will need to do one of the two following data work before training your model on the cloud.

1. Upload your locally stored data to a cloud storage to continue training on the cloud

OR

2. Load data from an existing cloud storage in a notebook to interactively prepare data and train

**In this tutorial, we are going to cover how to access and explore data already stored in a cloud storage.** If you want to learn how to upload your local data, go to [article name]. 

## PREP - DISCUSSION: if we are going to teach how to overwrite aka clean up data, we need the data to be actually in the user's storage instead of merely providing a sample data set link. However, it means we are not cherry picking "access and explore the data already stored in the cloud". We will have to cover the datastore/asset creation.

## Go to Studio Notebook interface
- Go to ml.azure.com
- Click Notebook
- Create a new notebook

## Connect to workspace
Run the authentication script in the notebook. This is to [reason why]

```python
# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<AML_WORKSPACE_NAME>",
)
```

## Install required libraries
To access data in a cloud storage, Azure Machine Learning supports `fasspec - filesystem interfaces for python`. This makes loading data files in a cloud storage simple and intuitive. You'll need to install `panda fsspec` and `mltable` libraries. Installing the libraries in your development environment would allow you to do X, Y, Z.

Q: do we want mltable for day 1?

```bash
!pip install -U azureml-fsspec mltable
```

## Run python script to access data
If you already know the location of the data you want to load in a notebook, the following code will do the job. Let's try accessing the credit card default data in an azure storage (original dataset is from UCI, proper citation goes here). If you want to learn more about how to load data from different cloud storages (ex. Databrick), go to [article name].



```python
import pandas as pd

#read data in the specified location
df = pd.read_csv("https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv") 

#return the first 5 row of the dataset
df.head() 

```

## Preview the data
We need to build the above script to do more. What should we put?

Explain the concept + instruction/code snippet

Note: Use data dictionary to explain what user is looking at in this preview stage
#### Data dictionary

The data contains 23 variables explanatory variables and 1 response variable, as described in the Table below:

|Column Name(s) | Variable Type  |Description  |
|---------|---------|---------|
|X1     |   Explanatory      |    Amount of the given credit (NT dollar): it includes both the individual consumer credit and their family (supplementary) credit.    |
|X2     |   Explanatory      |   Gender (1 = male; 2 = female).      |
|X3     |   Explanatory      |   Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).      |
|X4     |   Explanatory      |    Marital status (1 = married; 2 = single; 3 = others).     |
|X5     |   Explanatory      |    Age (years).     |
|X6-X11     | Explanatory        |  History of past payment. We tracked the past monthly payment records (from April to September  2005). -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.      |
|X12-17     | Explanatory        |  Amount of bill statement (NT dollar) from April to September  2005.      |
|X18-23     | Explanatory        |  Amount of previous payment (NT dollar) from April to September  2005.      |
|Y     | Response        |    Default payment (Yes = 1, No = 0)     |


## Process the data
Continue to build upon the previous step. What should we put?

Explain the concept + instruction/code snippet that runs data prep

Probably need something like this, but do we want to showcase MLflow?

```python
###################
#<prepare the data>
###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    credit_df = pd.read_csv(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )
####################
#</prepare the data>
####################
```

OR

```python
# read in data again, this time using the 2nd row as the header
df = pd.read_csv(data_asset.path, header=1)
# rename column
df.rename(columns={'default payment next month': 'default'}, inplace=True)
# remove ID column
df.drop('ID', axis=1, inplace=True)

# write file to filesystem
df.to_parquet('./cleaned-credit-card.parquet')
```

## Versioning
Pulled the below code from Sam's PR but we will need to tweak + explain the concept / what code does a bit more

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = './cleaned-credit-card.parquet'

# define data asset and use tags to make it clear the asset can be used in training
my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="Default of credit card clients data.",
    name="credit-card",
    version="2",
    tags={
        "training_data": "true",
        "format": "parquet"
    }
)

# create data asset
ml_client.data.create_or_update(my_data)
```

## Register data to be used for training
Pulled this from Sam's PR but we don't have the permission to write over


```python
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# get a handle for your AzureML workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# get a handle of the data asset and print the URI
data_asset_v1 = ml_client.data.get(name="credit-card", version="1")
data_asset_v2 = ml_client.data.get(name="credit-card", version="2")
print(f'V1 Data asset URI: {data_asset_v1.path}')
print(f'V2 Data asset URI: {data_asset_v2.path}')

v1df = pd.read_csv(data_asset_v1.path)
print(v1df.head(5))

v2df = pd.read_parquet(data_asset_v2.path)
print(v2df.head(5))
```
