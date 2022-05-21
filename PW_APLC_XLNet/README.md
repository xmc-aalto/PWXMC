Pytorch implementation for propensity-weighted variants of APLC_XLNet proposed in [Convex Surrogates for Unbiased Loss Functions in Extreme Classification With Missing Labels](https://dl.acm.org/doi/pdf/10.1145/3442381.3450139)

The code is adapted from the source code of APLC_XLNet [1]


## Requirements
* Linux
* Python ≥ 3.6
    ```bash
    # We recommend you to use Anaconda to create a conda environment 
    conda create --name aplc_xlnet python=3.6
    conda activate aplc_xlnet
    ```
* PyTorch ≥ 1.4.0
    ```bash
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
* Other requirements:
    ```bash
    pip install -r requirements.txt
    ```
## Prepare Data

### preprocessed data

Download the preprocessed datasets from [Google Drive](https://drive.google.com/drive/folders/1bRLrc8N3ukzAVn9zyTqr0IqP3fWJUYAt?usp=sharing) and save them to `data/`

### preprocess the custom dataset
1. Create `train.csv` and `dev.csv`. Reference our preprocessed dataset for the format of the CSV file
2. Create `labels.txt`. Labels should be sorted in descending order according to their frequency
3. Count the number of positive labels of each sample, select the largest one in all samples, and assign it to the hyperparameter `--pos_label`
4. Add the `dataset name` into the dictionary `processors` and `output_modes` in the source file `utils_multi_label.py`
5. Create the bash file and set the hyperparameters in `code/run/`

### raw text 
- For dataset EURlex, the raw text is from the [website](http://www.ke.tu-darmstadt.de/resources/eurlex/eurlex.html)
- For dataset Wiki500k, the raw text is from [Google Drive](https://drive.google.com/drive/folders/1KQMBZgACUm-ZZcSrQpDPlB6CFKvf9Gfb)
- For dataset Wiki10, AmazonCat and Amazon670k, the raw texts are from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html)

## Run the source code
### Training and evaluation
Run the commands
- For dataset EURlex:     `bash ./run/eurlex_pw.bash`
- For dataset Wiki10:     `bash ./run/wiki10_pw.bash`
- For dataset AmazonCat:  `bash ./run/amazoncat_pw.bash`
- For dataset Wiki500k:   `bash ./run/wiki500k_pw.bash`
- For dataset Amazon670k: `bash ./run/amazon670k_pw.bash`

The above scripts are for the PW variant of the proposed loss functions. For PW-cb, please set --reweighting to PW-cb.


## References
[1] Ye et al., [Pretrained Generalized Autoregressive Model with Adaptive Probabilistic Label Clusters for Extreme Multi-label Text Classification](http://arxiv.org/abs/2007.02439), ICML 2020

[2] Qaraei et al., [Convex Surrogates for Unbiased Loss Functions in Extreme Classification With Missing Labels](https://dl.acm.org/doi/pdf/10.1145/3442381.3450139), WWW 2021
