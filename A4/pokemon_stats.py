#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:35:59 2021

@author: ayujprasad
"""

import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def load_data(filepath):
    dataset = []
    with open(filepath, newline='', encoding='utf-8') as readfile:
        reader = csv.DictReader(readfile)
        row_num = 1
        for row in reader:
            if row_num > 20:
                break
            del row['Generation']
            del row['Legendary']
            row['HP'] = int(row['HP'])
            row['Attack'] = int(row['Attack'])
            row['Defense'] = int(row['Defense'])
            row['Sp. Atk'] = int(row['Sp. Atk'])
            row['Sp. Def'] = int(row['Sp. Def'])
            row['Speed'] = int(row['Speed'])
            row['#'] = int(row['#'])
            row['Total'] = int(row['Total'])
            dataset.append(row)
            row_num += 1
    return dataset


def calculate_x_y(stats):
    return (int(stats["Attack"]) + int(stats["Sp. Atk"]) + int(stats["Speed"]),
            int(stats["Defense"]) + int(stats["Sp. Def"]) + int(stats["HP"]))


def hac_calc_distance(dataset):
    m = len(dataset)
    euc_dist = list()
    
    # calculate the distances
    for i in range(m-1):
        for j in range(i+1, m):
            (x_i, y_i) = dataset[i]
            (x_j, y_j) = dataset[j]
            d = math.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
            euc_dist.append((i, j, d))
    
    # sort the points by their distances
    euc_dist = sorted(euc_dist, key = lambda x: x[2])
      
    return euc_dist

def hac_larger_cluster(m, c_list, cluster_of_i, cluster_of_j, ret, d, counter):
    cluster_with_i = c_list[cluster_of_i]
    cluster_with_j = c_list[cluster_of_j]
    len_i = len(cluster_with_i[1])
    len_j = len(cluster_with_j[1])
    if m+cluster_with_i[0] < m+cluster_with_j[0]:
        ret.append([m+cluster_with_i[0], m+cluster_with_j[0], d, len_i+len_j])
    else:
        ret.append([m+cluster_with_j[0], m+cluster_with_i[0], d, len_i+len_j])
    new_cluster = [counter, cluster_with_i[1]+cluster_with_j[1]]
    c_list.remove(cluster_with_i)
    c_list.remove(cluster_with_j)
    c_list.append(new_cluster)
    return c_list, ret

def hac_check1(m, c_list, cluster_of_j, ret, d, counter, i):
    if i < m+c_list[cluster_of_j][0]:
        ret.append([i, m+c_list[cluster_of_j][0], d, len(c_list[cluster_of_j][1])+1])
    else:
        ret.append([m+c_list[cluster_of_j][0], i, d, len(c_list[cluster_of_j][1])+1])
    new_cluster = [counter, c_list[cluster_of_j][1] + (i,)]
    c_list.remove(c_list[cluster_of_j])
    c_list.append(new_cluster)          
    return c_list, ret

def hac_check2(m, c_list, cluster_of_i, ret, d, counter, j):
    if m+c_list[cluster_of_i][0] < j:
        ret.append([m+c_list[cluster_of_i][0], j, d, len(c_list[cluster_of_i][1])+1])
    else:
        ret.append([j, m+c_list[cluster_of_i][0], d, len(c_list[cluster_of_i][1])+1])
    new_cluster = [counter, c_list[cluster_of_i][1] + (j,)]
    c_list.remove(c_list[cluster_of_i])
    c_list.append(new_cluster)
    return c_list, ret
    
def hac(dataset):
    g = len(dataset) -1
    while g >= 0:
        if math.isnan(dataset[g][0]) or math.isinf(dataset[g][0]) or math.isnan(dataset[g][1]) or math.isinf(dataset[g][1]):
            rem = dataset[g]
            dataset.remove(rem)
        g = g-1
       # (x_i, y_i) = dataset[i]
        #if (x_i == math.inf) or (x_i == -math.inf) or (x_i == float("nan")) or (x_i == float("inf")) or (x_i == float("-inf")):
         #   np.remove(dataset[i])
        #    #i = i+1
        #    g = g-1
        #if (y_i == math.inf) or (y_i == -math.inf) or (y_i == float("nan")) or (y_i == float("inf")) or (y_i == float("-inf")):
           # np.remove(dataset[i])
           # i = 1 + 1
           # g = g-1
    
    m = len(dataset)
    euc_dist = list() 
    ret = list() 
    c_list = list()

    euc_dist = hac_calc_distance(dataset)
    
    # track cluster row (used for merging)
    counter = 0

    for row in range(len(euc_dist)):
        # First cluster
        if len(ret) == 0:
            cluster_first = euc_dist.pop(0)
            ret.append([cluster_first[0], cluster_first[1], cluster_first[2], 2])
            c_list.append([counter, (cluster_first[0], cluster_first[1])])
            counter += 1
        else:
            next_cluster = euc_dist.pop(0)
            i = next_cluster[0]
            j = next_cluster[1]
            d = next_cluster[2]

            cluster_of_i = -1
            cluster_of_j = -1
            for cluster in c_list:
                if (i in cluster[1] and j in cluster[1]):
                    cluster_of_i = c_list.index(cluster)
                    cluster_of_j = c_list.index(cluster)
                elif i in cluster[1]:
                    cluster_of_i = c_list.index(cluster)
                elif j in cluster[1]:
                    cluster_of_j = c_list.index(cluster)
            
            #tie-breaking
            if len(euc_dist) > 0 and euc_dist[0][2] == d and (cluster_of_i != -1 or cluster_of_j != -1): 
                continue

            # new Cluster
            if cluster_of_i == -1 and cluster_of_j == -1:
                if i < j:
                    ret.append([i, j, d, 2])
                else:
                    ret.append([j, i, d, 2])
                new_cluster = [counter, (i, j)]
                c_list.append(new_cluster)
                
            # both in same cluster
            elif cluster_of_i == cluster_of_j != -1:
                continue
            
            # both different larger clusters
            elif (cluster_of_i != cluster_of_j) and (cluster_of_i != -1) and (cluster_of_j != -1):
                c_list, ret = hac_larger_cluster(m, c_list, cluster_of_i, cluster_of_j, ret, d, counter)

            # Two checks =whether one is in a larger cluster and one is alone
            elif cluster_of_i == -1 and cluster_of_j != -1:
                c_list, ret = hac_check1(m, c_list, cluster_of_j, ret, d, counter, i)
                
            elif cluster_of_i != -1 and cluster_of_j == -1:
                c_list, ret = hac_check2(m, c_list, cluster_of_i, ret, d, counter, j)
                
            counter += 1

        if len(ret) == m-1:
            break

    return np.array(ret)


def random_x_y(m):
    count = 0
    ret = list()
    
    while count < m:
        
        x_point = random.randint(1,359)
        y_point = random.randint(1,359)
        coord = (x_point, y_point)
        ret.append(coord)
        count+=1
            
    return ret

def imshow_hac(dataset):
    toPlot = hac(dataset)
    
    for i in range(len(toPlot)):
        x = i[0]
        y = i[1]
        plt.scatter(x,y)
        plt.pause(0.1)
    plt.show()    
