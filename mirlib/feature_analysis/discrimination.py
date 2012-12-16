
import numpy as np
import kmeans

class discriminator:

    def __init__(self, k):
        self.k = k

    def Within_Class_Scatter_Matrix(self, data, classLabels):
        L = len(data)
        resultMatrix = np.zeros([data.shape[1], data.shape[1]])
        for k in range(self.k):
            # Get the number occurences of class k
            L_k = len((classLabels == k).nonzero()[0])

            # Covariance Matrix for samples in class k
            #print data[(classLabels == k).nonzero()[0]].T
            C_k = np.cov(data[(classLabels == k).nonzero()[0]].T)

            resultMatrix += (L_k / np.float(L)) * C_k
        return resultMatrix

    def Between_Class_Scatter_Matrix(self, data, classLabels):
        L = len(data)
        resultMatrix = np.zeros([data.shape[1], data.shape[1]])
        mu = np.mean(data, axis=0) #global mean

        for k in range(self.k):
            # Get number of occurences of k
            L_k = len((classLabels == k).nonzero()[0])
            mu_k = np.mean(data[ (classLabels == k).nonzero()[0] ], axis=0) # class mean

            mu_diff = mu_k - mu
            resultMatrix += (L_k / np.float(L)) * np.outer(mu_diff,mu_diff)

        return resultMatrix

    def Discriminitive_Power(self, data, classLabels):
        Sw = self.Within_Class_Scatter_Matrix(data, classLabels)
        print Sw
        print "SW END"
        Sb = self.Between_Class_Scatter_Matrix(data, classLabels)
        print "SB END"

        J0 = np.trace(Sw) / np.trace(Sb)
        print "J0 END"
        return J0

def Disc_Power(data, k, classes):
    Dtor = discriminator(k)
    return Dtor.Discriminitive_Power(data, classes)

        
def main():
    testData = np.random.randn(200, 10)
    k = 4

    centroids, distortion = kmeans.scipy_kmeans(testData, k)
    classes, dist = kmeans.scipy_vq(testData, centroids)

    Dtor = discriminator(k)
    Sw = Dtor.Within_Class_Scatter_Matrix(testData, classes)
    print Sw, Sw.shape

    Sb = Dtor.Between_Class_Scatter_Matrix(testData, classes)
    print Sb, Sb.shape

    J0 = Dtor.Discriminitive_Power(testData, classes)
    print J0, J0.shape

if __name__ == "__main__":
    main()
