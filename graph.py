import matplotlib.pyplot as plt 
import numpy as np
import os

def plot_graph(y,z,name_y,name_z,similarity):
    dirpath = "C://Users//Kumar Abhijeet//comptt/LDA//plots//"
    x = np.arange(21) + 1
    x1 = x - 0.2
    ax = plt.subplot(111)
    w = 0.3
    ax.bar(x1 ,y ,width = w, color='b',align = 'center',label=name_y)
    ax.bar(x ,z ,width = w, color='r',align = 'center',label=name_z)
    plt.xticks(x)
    plt.xlabel("Topics")
    plt.ylabel("Probability Score")
    plt.title("Document-Topic Probablity Similarity Distribution\n Score = " + str(similarity))
    ax.legend()
    plt.savefig(dirpath  + name_y +"_" + name_z + ".png")
    plt.gcf().clear()

#y = np.random.randint(1,20,21)
#z = np.random.randint(1,20,21)
#plot_graph(y,z,'a','b')