def writeNewDict(dict,index):
	# constructs specific dictionary from master dictionary
	mydict = {}
	for key, value in dict.items():
		if isinstance(value,list):
			if len(value) == 1: # a list of length 1
				valuenew = value[0]
			else: # list of length n
				valuenew = value[index]
		else: # not a list, just a singleton
			valuenew = value
		mydict[key] = valuenew
	return mydict
    
def dictUnion(listofdicts): # possible overwriting of old entries, if duplicate keys
    dictres = {}
    for dict in listofdicts:
        dictres.update(dict)
    return dictres