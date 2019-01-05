def numOfNones(List):
    num = 0
    for e in List:
        if e is None: num+=1
    return num

def removeNones(List=[]):
    removeIndexs = []
    for i in range(len(List)):
        if List[i] is None:
            removeIndexs.append(i)
    return np.delete(List, removeIndexs)