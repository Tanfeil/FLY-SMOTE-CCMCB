# FLY-SMOTE: Re-balancing the non-IID IoT Edge Devices Data in Federated Learning System

In recent years, the data available from IoT devices have increased rapidly. Using a machine learning solution to detect faults in these devices requires the release of device data to a central server. However, these data typically contain sensitive information, leading to the need for privacy-preserving distributed machine learning solutions, such as federated learning, where a model is trained locally on the edge device, and only the trained model weights are shared with a central server. Device failure data are typically imbalanced, i.e., the number of failures is minimal compared to the number of normal samples. Therefore, re-balancing techniques are needed to improve the performance of a machine learning model. In this paper, we present FLY-SMOTE, a new approach to re-balance the data in different non-IID scenarios by generating synthetic data for the minority class in supervised learning tasks using a modified SMOTE method. Our approach takes $k$ samples from the minority class and generates $Y$ new synthetic samples based on one of the nearest neighbors of each $k$ sample. An experimental campaign on a real IoT dataset and three well-known public datasets show that the proposed solution improves the balance accuracy without compromising the model's accuracy.

## The data used in this project:
*  [Adult](https://archive.ics.uci.edu/ml/datasets/adult)
* [Compass](https://www.kaggle.com/datasets/danofer/compass)
* [Bank](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Code
The code is divided as follows:

* The main.py python file contains the necessary code to run an experiement.
* The FlySmote.py contains the necessary functions to apply fly-smote re-balancing method.
* the NNModel.py contains the neural network model.
* The ReadData.py file contains the necessary functions to read the datasets.

To run a model on one dataset you should issue the following command:

```bash
python main.py -f <dataname> -d <data file name> -k <samples from miniority> -r <ratio of new samples>
```

## Prerequisites
The python packages needed are:
* numpy
* pandas
* sklearn
* scipy
* matplotlib
* tensorflow
* keras

## Reference
If you re-use this work, please cite:
```
@ARTICLE{9800764,
  author={Younis, Raneen and Fisichella, Marco},
  journal={IEEE Access}, 
  title={FLY-SMOTE: Re-Balancing the Non-IID IoT Edge Devices Data in Federated Learning System}, 
  year={2022},
  volume={10},
  number={},
  pages={65092-65102},
  doi={10.1109/ACCESS.2022.3184309}}
```
