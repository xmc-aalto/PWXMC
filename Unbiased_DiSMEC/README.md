C++ implementation for unibased variants of DiSMEC in [Convex Surrogates for Unbiased Loss Functions in Extreme Classification With Missing Labels](https://dl.acm.org/doi/pdf/10.1145/3442381.3450139)

The code is adapted from the source code of DiSMEC [2]


## CONTENTS

There are following directories:
1. ./dismec contains the dismec code 

2. ./eurlex consists of data for eurlex-4k downloaded from XMC repository

3. ./prepostprocessing consists of Java code for (a) pre-processing data to get into tf-idf format and remapping labels and features, and (b) Evaluation of propensity scored precision@k and nDCG@k corresponding to the prediction results.


## Data Pre-processing (in Java)
0. Download the eurlex dataset from XMC repository, and remove the first line from the train and test files downloaded, call them train.txt and test.txt

1. Change feature ID's so that they start from 1..to..number_of_features, using the code provided in FeatureRemapper.java using the following command
```bash
javac FeatureRemapper.java
java FeatureRemapper ../eurlex/train.txt ../eurlex/train-remapped.txt ../eurlex/test.txt ../eurlex/test-remapped.txt
```

2. Convert to tf-idf format using the code in file TfIdfCalculator.java
```bash
javac TfIdfCalculator.java
java TfIdfCalculator ../eurlex/train-remapped.txt ../eurlex/train-remapped-tfidf.txt ../eurlex/test-remapped.txt ../eurlex/test-remapped-tfidf.txt
```

3. Change labels ID's so that they also start from 1..to..number_of_labels, using the code provided in LabelRelabeler.java 
```bash
javac LabelRelabeler.java 
java LabelRelabeler ../eurlex/train-remapped-tfidf.txt ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/test-remapped-tfidf.txt ../eurlex/test-remapped-tfidf-relabeled.txt ../eurlex/label-mappings.txt
```

## Building DiSMEC

Just run make command in the ../dismec/ directory. This will build the train and predict executable


## Training model with (unbiased) DiSMEC

```bash
mkdir ../eurlex/models # make the directory to write the model files

REW_TYPE=1 # can be 0, 1, or 2, corresponding to normal, PW, and PW-cb training, respectively)

../dismec/train -s 2 -B 1 -i 1 -r $REW_TYPE -a 0.55 -b 1.5 -l 3786 -z 1000 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/1.model
../dismec/train -s 2 -B 1 -i 2 -r $REW_TYPE -a 0.55 -b 1.5 -l 3786 -z 1000 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/2.model
../dismec/train -s 2 -B 1 -i 3 -r $REW_TYPE -a 0.55 -b 1.5 -l 3786 -z 1000 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/3.model
../dismec/train -s 2 -B 1 -i 4 -r $REW_TYPE -a 0.55 -b 1.5 -l 3786 -z 1000 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/4.model
```

## Predicting with (unibased) DiSMEC in parallel

Since the base Liblinear code does not understand the comma separated labels. We need to zero out labels in the test file, and put that in a separate file (called GS.txt) consisting of only the labels.
```bash
javac LabelExtractor.java
java LabelExtractor ../eurlex/test-remapped-tfidf-relabeled.txt ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/GS.txt

 
mkdir ../eurlex/output # make the directory to write output files
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/1.model ../eurlex/output/1.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/2.model ../eurlex/output/2.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/3.model ../eurlex/output/3.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/4.model ../eurlex/output/4.out
```

## Performance evaluation (in Java)

Computation of Precision@k and nDCG@k for k=1,3,5
Now, we need to get final top-1, top-3 and top-5 from the output of individual models. This is done by the following:

****** IMPORTANT : Change the number of test points in DistributedPredictor.java (at line number 138) based on number of test points in the datasets ******

```bash
mkdir ../eurlex/final-output
javac PropensityComputer.java
java PropensityComputer ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/inv_prop.txt 15511 0.55 1.5

javac DistributedPredictor.java
java DistributedPredictor ../eurlex/output/ ../eurlex/final-output/top1.out ../eurlex/final-output/top3.out ../eurlex/final-output/top5.out ../eurlex/inv_prop.txt ../eurlex/final-output/top1-prop.out ../eurlex/final-output/top3-prop.out ../eurlex/final-output/top5-prop.out

javac MultiLabelMetrics.java
java MultiLabelMetrics ../eurlex/GS.txt ../eurlex/final-output/top1.out ../eurlex/final-output/top3.out ../eurlex/final-output/top5.out ../eurlex/inv_prop.txt ../eurlex/final-output/top1-prop.out ../eurlex/final-output/top3-prop.out ../eurlex/final-output/top5-prop.out
```

## References
[1] R.-E. Fan et al., [LIBLINEAR: A library for large linear classification](https://dl.acm.org/doi/pdf/10.5555/1390681.1442794), Journal of Machine Learning Research 9(2008), 1871-1874.

[2] R. Babbar, B. Schölkopf, [DiSMEC: Distributed Sparse Machines for Extreme Multi-label Classification](https://dl.acm.org/doi/pdf/10.1145/3018661.3018741), WSDM 2017.

[3] M. Qaraei et al., [Convex Surrogates for Unbiased Loss Functions in Extreme Classification With Missing Labels](https://dl.acm.org/doi/pdf/10.1145/3442381.3450139), WWW (2021).
