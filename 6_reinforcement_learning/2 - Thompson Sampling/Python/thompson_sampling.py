# Thompson Sampling (more powerful than Upper Confidence Bound)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling (Probabilistic Solutions)
import random
N = 10000 # Number of users that we are showing ads consecutivly. We can try with all data to be deterministic, but the idea is try over the march, so with few data as 10% of total
d = 10 # Number of Ads to show
ads_selected = []

#For each list we should have a variable when and Ad was selected and when not
numbers_of_rewards_1 = [0] * d #for each ad, should increase if ad was selected
numbers_of_rewards_0 = [0] * d #for each ad, should increase if ad was not selected
total_reward = 0
for n in range(0, N): # For loop users
    ad = 0 # Index of the Ad that will be selected, starting from 0
    max_random = 0 # Max value drawn randomly from the distribution
    for i in range(0, d): # For loop ads
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad) # after loop ads, selec the add with higher probability of higher return
    reward = dataset.values[n, ad] # Selected ad of specific row, to check if it was selected or not
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward # updating value of total_reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()