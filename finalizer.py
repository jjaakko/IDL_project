import numpy as np
import pandas as pd

def add_missing(predictions):
    """Adds zeroes for columns of redundant labels and writes the result into a a file 'pred_numpy.npy'.
    
    Args:
    predictions: Numpy array of size n * 103, holding the predicted labels as multi-hot binary vectors, with n as sample size.
    """
    n = np.shape(predictions)[0]
    final = np.hstack((
        np.zeros((n,11)),
        predictions[:,:60],
        np.zeros((n,9)),
        predictions[:,60:75],
        np.zeros((n,1)),
        predictions[:,75:],
        np.zeros((n,2))
    ))

    np.save(f"pred_numpy", final)

# Create random label data to test the 'add_missing' function
def create_random():
    pred = np.zeros((100,103))
    nrows = (len(pred[:,0]))
    for i in range(nrows):
        random_indexes = np.random.choice(103, size=(8), replace=False)
        for x in random_indexes:
            pred[i][x] += np.random.randint(2)

    return (pred)

prediction = create_random()
add_missing(prediction)

# Perform a sanity check by counting the number of positives in each column (Columns 0-10,71-79,95 and 124-125 should be zero).
sanity_check = np.load("pred_numpy.npy")
zero_idx = [0,1,2,3,4,5,6,7,8,9,10,71,72,73,74,75,76,77,78,79,95,124,125]
print("col\tn_positives")
for i in range(126):
    print(i,"\t",np.sum(sanity_check[:,i]))

# Another sanity check to see if the sum of positives in redundant columns is zero    
print("\nThis should be zero:",np.sum(sanity_check[:,zero_idx]))