from collections import Counter 
import collections 
import math 

def entropy(class_probabilities):
	"given a list of class probabilities, compute the entropy"
	return sum(-p*math.log(p,2) 
		       for p in class_probabilities if p)


def class_probabilities(labels):
	total_count=len(labels)
	return [count/total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
	labels=[label for _, label in labeled_data]
	probabilities = class_probabilities(labels)
	return entropy(probabilities)

def partition_entropy(subsets):
	"find the entropy from this partition of data into subsets \
	subsets is a list of lists of labeled data"
	total_count = sum( len( subset) for subset in subsets) 
	return sum( data_entropy( subset) * len( subset) / total_count for subset in subsets )


def partition_by(inputs,attribute):
	"each input is a pair , returns a dict : attribute_value --> input"
	groups = collections.defaultdict(list)
	for input in inputs:
		key = input[0][attribute]
		groups[key].append(input)
	return groups 

def partition_entropy_by(inputs, attribute):
	partitions = partition_by (inputs, attribute)
	return partition_entropy(partitions.values())


inputs = [({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False), 
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False), 
({'level':'Mid', 'lang':' Python', 'tweets':'no', 'phd':'no'}, True), 
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True), 
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True), 
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False), 
({'level':'Mid', 'lang':' R', 'tweets':'yes', 'phd':'yes'}, True),
 ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
  ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True), 
  ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
   ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True), 
   ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
 ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
 ]

for key in ['level','lang','tweets','phd']:
	print(key,partition_entropy_by(inputs,key))