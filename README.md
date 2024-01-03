
# PrivTC - Implementation

### Reference: [Paper](https://github.com/Lonelypheonix/PrivTC-Implementation/blob/main/Paper/Collecting_Individual_Trajectories_under_Local_Differential_Privacy.pdf)

```
J. Yang, X. Cheng, S. Su, H. Sun and C. Chen, 
"Collecting Individual Trajectories under Local Differential Privacy,"
2022 23rd IEEE International Conference on Mobile Data Management (MDM), Paphos, Cyprus, 2022, pp. 99-108, 
doi: 10.1109/MDM55031.2022.00035.
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9861201&isnumber=9861093
```

In this paper, we tackle the problem of collecting individual trajectories under LDP.

- The problem is :  **Given a large number of users, each of which has a trajectory in the given geospatial domain, an untrusted aggregator aims at collecting trajectories from these users while satisfying LDP.**
- Solution :  **Hidden Markov Model (HMM)**  to approximate the overall distribution of users’ trajectories in a locally differentially private manner and generate a synthetic trajectory dataset as a surrogate for the original one from the learned model.
- However, based on this idea, designing an effective approach which enables the aggregator to obtain the synthetic trajectory dataset with high utility is nontrivial due to the following two technical challenges.
    - **First Challenge:**    The problem arises from the continuous nature of latitude and longitude coordinates. If we were to directly apply a Hidden Markov Model (HMM) to the original trajectories, it would result in an infinite number of possible paths when simulating movement from one location to another. This is due to the continuous nature of geographic coordinates, making it challenging to discretize and model the transitions effectively in the HMM framework.
    - **Second challenge:** how to accurately learn the HMM under LDP.
- The General solution would be
    - The idea is to ensure Local Differential Privacy (LDP) in the Baum-Welch algorithm, which is commonly used in Hidden Markov Models. However, applying LDP directly to the algorithm would involve numerous interactions between the data aggregator and users. This could result in significant delays in the network and add a considerable amount of noise to achieve privacy. As a result, this approach becomes impractical in real-world scenarios due to the high network latency and excessive noise introduced.


### Solution from the paper - PrivTC

- In PrivTC , we design a locally differentially private grid construction method called PrivAG,
- **PrivAG :** it will adaptively partition the given geospatial domain into a reasonable grid. According to the constructed grid, the users can discretize their trajectories to bound the number of probabilities.
- Therefore, **the first challenge is solved.**
- Then they proposed a locally differentially private spectral learning method called PrivSL
- **PrivSL :** It employs spectral learning to learn the HMM from users’ discretized trajectories.
- Since PrivSL requires the aggregator to interact with each user only once, **it overcomes the second challenge.**


### PrivTC Procedure:

- At the beginning , the aggregator randomly divides users into two groups **U1** and **U2**.
- Then, the aggregator uses **PrivAG** to partition the given geospatial domain into a **grid G** by interacting with **U1** and broadcasts the **grid G** to each user in **U2**.
- After that, the aggregator uses **PrivSL** to learn the **HMM** from the **U2**’s trajectories discretized according to **G**.
- Finally, the aggregator **generates a synthetic trajectory dataset** from the learned HMM.


## Implementation

Please Check the example code to understand how it works [Example python notebook](https://github.com/Lonelypheonix/PrivTC-Implementation/blob/main/Privtc_Reproduction%20.ipynb). 

## Content
The code is implemented using Python3.7. The descriptions of files are in the following:

```
PrivTC 
|
|- algorithm_PrivGR_PrivSL.py: PrivTC algorithm
|  |- PrivGR.py: PrivAG algorithm
|  |- PrivSL.py: PrivSL algorithm
|  |- choose_granularity.py: set the granularities by guideline
|  |- consistency_method.py: norm_sub
|  |- frequecy_oracle.py: LDP categorical frequency oracles including OUE and OLH
|  |- grid_generate.py: define the cell in the grid
|  |- parameter_setting.py: define the parameters
|
|  |- utility_metric_query_avae.py: calculate Query MAE
|  |- utility_metric_query_FP.py: calculate FP Similarity
|  |- utility_metric_query_length_error.py: calculate Distance Error
|  
|  |- Privtc.py : PrivTC algorithm for epsilon value 1
|  |- Privtc_eps.py : PrivTC algorithm for a range of epsilon values 
|  

```

The [Privtc.py](https://github.com/Lonelypheonix/PrivTC-Implementation/blob/main/Code/Privtc.py) is used to illustrate how to run PrivTC algorithm for epsilon value 1.

The [Privtc_eps.py](https://github.com/Lonelypheonix/PrivTC-Implementation/blob/main/Code/Privtc_eps.py) is used to illustrate how to run PrivTC algorithm for arange of epsilon values.

## Note:
If you are using google collab then use 
```
%run Privtc.py
```
This is used to display the graph in Inline mode.

If you are using the command line then use:

```
python3 Privtc.py
```
If there are any errors, 
- Check the location of dataset file in the code.
- Check if you have all the packages installed.



## Author

- [@Pavan Kumar J](https://github.com/Lonelypheonix)


## Feedback

If you have any questions, please reach out to me at pavankumarj.cy@gmail.com

