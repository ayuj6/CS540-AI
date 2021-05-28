#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:31:08 2021

@author: ayujprasad
"""
import heapq

def print_succ(state):
    successors = get_successor(state)
    for i in range(len(successors)):
        hue_val = get_Manhattan_dis(successors[i], [1,2,3,4,5,6,7,8,0])
        successor = successors[i]
        print(str(successor) + str(' h=') + str(hue_val))

def get_parent(info):
    if info[4] != -1:
        get_parent(info[4])
        print(f'{info[1]} h={info[3]} moves: {info[2]}')
    else:
        print(f'{info[1]} h={info[3]} moves: {info[2]}')
        return
        
def solve(cur_state):
    closed = []
    open_pq = []
    final = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    g_score = 0
    state_info = [0, cur_state, 0, 0, -1]
    state_info[0]= get_Manhattan_dis(cur_state, final) + g_score
    state_info[3]= get_Manhattan_dis(cur_state, final)
    min_item = state_info
    while not (min_item) == None:
        if cur_state == final:
            get_parent(min_item)
            return
        successors = get_successor(cur_state)
        g_score = min_item[2] + 1 
        for suc_state in successors:
            state_info = [0, suc_state, g_score, 0, min_item]
            state_info[0] = get_Manhattan_dis(suc_state, final) + g_score
            state_info[3] = get_Manhattan_dis(suc_state, final)
            heapq.heappush(open_pq, state_info)
        
        min_item = heapq.heappop(open_pq)
        cur_state = min_item[1]
        if cur_state in closed:
            min_item = heapq.heappop(open_pq)
            cur_state = min_item[1]
        closed.append(cur_state)

  
''' Original Author: Ajinkya Sonawane
    Source:https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
    The following code mimics the code on the site, and is modified to meet the code above.
'''
def get_successor(state):
    row,column = findEmpty(state,0)
    neighbor_list = [[row,column-1],[row,column+1],[row-1,column],[row+1,column]]
    valid_succ = []
    for i in neighbor_list:
        successors = shuffle(state,row,column,i[0],i[1])
        if successors is not None:
            successor = []
            for j in range(3):
                for k in range(3):
                    successor.append(successors[j][k])
            valid_succ.append(successor)
    sorted_succ = sorted(valid_succ)
    return sorted_succ
        
''' Original Author: Ajinkya Sonawane
    Source:https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
    The following code mimics the code on the site, and is modified to meet the code above.
'''
def shuffle(state,x1,y1,x2,y2):
    if x2 >= 0 and x2 < 3 and y2 >= 0 and y2 < 3:
        successor = []
        successor = copy(state)
        temp = successor[x2][y2]
        successor[x2][y2] = successor[x1][y1]
        successor[x1][y1] = temp
        return successor
    else:
        return None

''' Original Author: Ajinkya Sonawane
    Source:https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
    The following code mimics the code on the site, and is modified to meet the code above.
'''
def copy(state):
    ret = []
    for i in range(3):
        ret.append([0 for j in range(3)])
    index = 0
    for i in range(3):
        for j in range(3):
            ret[i][j] = state[index]
            index += 1
    return ret
    
''' Original Author: Ajinkya Sonawane
    Source:https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
    The following code mimics the code on the site, and is modified to meet the code above.
'''    
def findEmpty(state,empty):
        temp = copy(state)
        for i in range(3):
            for j in range(3):
                if temp[i][j] == empty:
                    return i,j
                
''' Original Author: AcrobatAHA
    Source:https://github.com/AcrobatAHA/How-to-solve-an-8-puzzle-problem-using-A-Algorithm-in-python-/blob/master/Heuristics%20for%20the%208-puzzle.py
    The following code is obtained the inspiraion from source linked above.
'''
def get_Manhattan_dis(oneD_cur_state, oneD_goal_state):
    dis = 0
    cur_state = copy(oneD_cur_state)
    goal_state = copy(oneD_goal_state)
    for i in range(len(cur_state)):
        for j in range(len(cur_state)):
            if cur_state[i][j] == 0:
                continue
            elif (goal_state[0][0] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-0))
            elif (goal_state[0][1] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-1))
            elif (goal_state[0][2] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-2))
            elif (goal_state[1][0] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-0))            
            elif (goal_state[1][1] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-1))
            elif (goal_state[1][2] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-2))
            elif (goal_state[2][0] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-0))
            elif (goal_state[2][1] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-1))   
            elif (goal_state[2][2] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-2))
    return dis
