import matplotlib.pyplot as plt
import numpy as np
import sys

    
def k_Means(pixels, z, out_fname):
    f = open(out_fname,"w")
    clusters_num = z.shape
    old_z = np.zeros(clusters_num)
    pixelsArr = np.array(pixels)
    pixels_num = pixelsArr.shape
    #making another copy to compare with in each iter
    new_z = np.copy(z)
    distances = np.zeros((pixels_num[0], clusters_num[0])) 
    iter = 0
    convergence = False
    while iter < 20 and not convergence:
        # calcuate the distance from each pixel
        for i in range(clusters_num[0]):
            distances[:, i] = np.linalg.norm(pixelsArr - new_z[i], axis=1)
        # Assign each data point to the closest cluster by calculating its distance to each centroid
        clusters = np.argmin(distances, axis=1)
        #put the current centroid in old_z so we can compare with after changing them
        old_z = np.copy(new_z)
        # Calculate mean for every cluster and update the centroids accordingly
        for i in range(clusters_num[0]):
                if pixelsArr[clusters == i].shape[0] > 0:
                    new_z[i] = np.mean(pixelsArr[clusters == i], axis=0).round(4)
                else:
                    new_z[i] = old_z[i]
        #checking if there is a change to continue
        convergence = np.linalg.norm(new_z - old_z) == 0
        f.write(f"[iter {iter}]:{','.join([str(i) for i in new_z])}\n")
        iter += 1
    f.close()



    
def main():
    image_fname,centroids_fname,out_fname = sys.argv[1],sys.argv[2],sys.argv[3]
    z = np.loadtxt(centroids_fname) #load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float)/255
    #Reshape the image
    pixels = pixels.reshape(-1,3)
    

    #call k_means
    k_Means(pixels, z, out_fname)


if __name__ == "__main__":
    main()
