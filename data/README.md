## Setting up the data

The benchmarking scripts expects 2 files to be present in `data/huffpost`.

`data/huffpost/train_all.csv` : training data
`data/huffpost/test.csv` : testing data

To setup the data for benchmarking under these requirements, do the following:

1. Download the data from https://www.kaggle.com/datasets/rmisra/news-category-dataset and unzip it.  This should produce a file called `News_Category_Dataset_v2.json` which we will need to split and save into the required files.
> *Please see this data set's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.*
   
2. Use the `prepare_data.py` script to generate the `huffpost/train_all.csv` and `huffpost/test.csv` files for benchmarking.  This script expects `News_Category_Dataset_v2.json` to be present in the same directory.

    `conda activate doc_class_intel`

    `cd data && python process_data.py`
