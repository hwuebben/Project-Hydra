import numpy as np

class feature_extraction:

    def calcVarX(pic):
        distribution = np.zeros(np.shape(pic)[0])
        for i in range(np.shape(pic)[0]):
            distribution[i] = np.sum(pic[i,:])
        distribution = distribution[np.nonzero(distribution)]
        distribution = np.divide(distribution,np.max(distribution))
        return np.std(distribution)

    def calcVarY(pic):
        distribution = np.zeros(np.shape(pic)[1])
        for i in range(np.shape(pic)[1]):
            distribution[i] = np.sum(pic[:,i])
        distribution = distribution[np.nonzero(distribution)]
        distribution = np.divide(distribution,np.max(distribution))
        return np.std(distribution)
