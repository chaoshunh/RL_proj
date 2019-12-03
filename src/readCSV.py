import os, sys, csv
import numpy as np 

pi = np.pi
parenpath = os.path.join(sys.path[0], '..')
def readCSV(state_size, action_size, filename=None):
    """
    The function is for reading dataset.
    """
    global parenpath
    memory = []
    with open(str(parenpath + '/assets/' + filename),'r') as csvfile:
        reader = csv.reader(csvfile)
        for oneline in reader:
            memory.append(map(float, oneline))
        
    for j in range(len(memory)):
        cur_row = memory[j]
        try:
            next_state = memory[j+1][:state_size]
        except:
            next_state = cur_row[:state_size]
        memory[j] = cur_row + next_state
        memory[j][6] /= 1000
        memory[j][-4] /= 1000
        memory[j][7] = memory[j][7] / 180 * pi
        memory[j][-3] = memory[j][-5] / 180 * pi

    memory = np.around(np.array(memory), decimals=4).tolist()
    return memory
