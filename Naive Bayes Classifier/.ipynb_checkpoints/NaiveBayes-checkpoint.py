import numpy as np


# ## get_att_vals
# * Takes in dataset
# * Returns a dict of attributes and their potential values as a list

def get_att_vals(data):
    att_vals = {}
    for att in range(len(data[0])):
        vals = np.unique(data[:, att])
        att_vals[att] = vals
    return(att_vals)

test_data = np.asarray([['a', 'c', 'b'], ['b', 'a', 'b']])
test_att_vals = get_att_vals(test_data)
assert(np.array_equal(test_att_vals[0],['a', 'b']))
assert(np.array_equal(test_att_vals[1],['a', 'c']))
assert(np.array_equal(test_att_vals[2],['b']))


# ## get_att_counts
# * Takes in a dict of attribute to values list, dataset as numpy 2d array, and smoothing as a bool)
# * Returns a nested dict, where each attribute maps to a dict of potential labels as keys and counts as values

def get_att_counts(att_vals, data, smoothing):
    counts = {}
    for att, vals in att_vals.items():
        att_col = data[:, att]
        val_counts = {}
        for val in vals:
            count = (att_col==val).sum()
            if smoothing: count += 1
            val_counts[val] = count
        counts[att] = val_counts
    return counts #counts[attribute][value] = the occurences of 'value' for given attribute

test_att_counts = get_att_counts(test_att_vals, test_data, smoothing=False)
assert(len(test_att_counts) == 3)
assert(test_att_counts[0]['a'] == 1)
test_smooth_att_counts = get_att_counts(test_att_vals, test_data, smoothing=True)
assert(test_smooth_att_counts[2]['b'] == 3)


# ## get_att_probs
# * Takes in the nested dict of attributes and value counts
# * Returns a nested dict of attributes and probabilities of each value

def get_att_probs(att_counts):
    att_probs = {}
    for attribute, counts in att_counts.items():
        val_probs = {}
        total = sum(counts.values())
        for value, count in counts.items():
            prob = float(count/total)
            val_probs[value] = prob
        att_probs[attribute] = val_probs
    return att_probs #att_probs[attribute][value] = prob of value occuring for given att

test_att_probs = get_att_probs(test_att_counts)
assert(len(test_att_probs) == 3)
assert(test_att_probs[0] == {'a': 0.5, 'b': 0.5})
assert(test_att_probs[2]['b'] == 1)


# ## get_intersect_prob
# * Takes in two attributes, and their corresponding values, as well as the dataset
# * Returns the probability of intersection of the two values for the given attributes

def get_intersect_prob(att_1, val_1, att_2, val_2, data):
    intersect = data[(data[:, att_1] == val_1) & (data[:, att_2] == val_2)]
    return float(len(intersect)/len(data))

test_intersect_prob = get_intersect_prob(0, 'a', 2, 'b', test_data)
assert(test_intersect_prob == 0.5)
test_intersect_prob = get_intersect_prob(0, 'b', 2, 'c', test_data)
assert(test_intersect_prob == 0)
test_intersect_prob = get_intersect_prob(1, 'c', 2, 'b', test_data)
assert(test_intersect_prob == 0.5)


# ## get_cond_probs
# * Takes in the nested dict of attributes and value porbabilities, attributes, and data
# * Returns the conditional probabilities of each x|y

def get_cond_probs(att_probs, atts, data): #calculate x|y
    probs = {}
    for label in atts[0]: #our 'x' values
        att_cond_probs = {}
        for att in range(1,len(atts)):
            val_probs = {}
            for val in atts[att]:
                if att_probs[att][val] == 0: prob = 0
                else:
                    prob = float(get_intersect_prob(0, label, att, val, data)/
                             att_probs[att][val])
                val_probs[val] = prob
            att_cond_probs[att] = val_probs
        probs[label] = att_cond_probs
    return probs #probs[label][att][val] = p(label|val)

test_cond_probs = get_cond_probs(test_att_probs, test_att_vals, test_data)
assert(test_cond_probs['a'][2]['b'] == 0.5)
assert(test_cond_probs['b'][1]['c'] == 0)
assert(test_cond_probs['b'][1]['a'] == 1)


# ## train 
# * Takes in data, the dict of attributes and values, and smoothing as a boolean
# * Returns the model which is a set of conditional probabilities

def train(data, att_vals, smoothing = True):
    att_counts = get_att_counts(att_vals, data, smoothing)
    att_probs = get_att_probs(att_counts)
    cond_probs = get_cond_probs(att_probs, att_vals, data)
    return(cond_probs)

test_model = train(test_data, test_att_vals)
assert(test_model['a'][2]['b'] == 0.5)
assert(test_model['a'][1]['c'] == 1)
assert(test_model['b'][1]['c'] == 0)


# ## get_prob
# * Takes in the model as a triple nested dict, instance as a list of attributes, and label we are calculating probability for
# * Returns the probability of that label for the given instance

def get_prob(model, instance, label):
    probs = []
    for att in range(len(instance)-1):
        val = instance[att]
        prob = model[label][att+1][val]
        probs.append(prob)
    return(np.prod(probs))

assert(get_prob(test_model, ['c','b'], 'a') == 1)
assert(get_prob(test_model, ['a','b'], 'a') == 0)
assert(get_prob(test_model, ['a','b'], 'b') == 1)


# ## normalize
# * Takes in results as a dict of label keys and probability values
# * Returns a dict of label keys and normalized probabilitiy values

def normalize(results):
    normalized = {}
    total = float(sum(results.values()))
    for label, prob in results.items():
        if(total == 0): normalized[label] = 0
        else: normalized[label] = float(prob/total)
    return normalized

assert(normalize({"0": 0.01, "1": 0.01}) == {"0": 0.5, "1": 0.5})
assert(normalize({"0": 0.01, "1": 0.03}) == {"0": 0.25, "1": 0.75})
assert(normalize({"0": 0.00, "1": 0.01}) == {"0": 0, "1": 1})


# ## classify_instance
# * Takes in a model as a triple nested dict and an instance as a list of attributes
# * Returns a tuple of the best label, and a dict of label/probability key-values

def classify_instance(model, instance):
    results = {}
    for label in model.keys():
        results[label] = get_prob(model, instance, label)
    results = normalize(results)
    best = max(results, key=results.get)
    return (best, results)

test_best, test_results = classify_instance(test_model, ['a','b'])
assert(test_best == 'b')
assert(test_results['a'] == 0)
assert(test_results['b'] == 1)


# ## classify
# * Takes in model as a triple nested dict, instances as a list of list of attributes, and labeled as a boolean
# * Returns a list of tuples of best labels and dict of labels to probabilities

def classify(model, instances, labeled=True):
    output = []
    for instance in instances:
        if labeled: instance = instance[1:]
        best, results = classify_instance(model, instance)
        output.append((best,results))
    return output

test_instances = [['c','b'], ['a','b']]
test_results = classify(test_model, test_instances, labeled=False)
assert(test_results[0][1]['b'] == 0)
assert(test_results[0][1]['a'] == 1)
labeled_test_instances = [['a','c','b'], ['a','a','b']]
labeled_test_results = classify(test_model, labeled_test_instances, labeled=True)
assert(test_results == labeled_test_results)


# ## evaluate
# * takes in the dataset as a 2d np array and results as a list of predicted labels
# * Returns the error rate

def evaluate(data, results):
    total = len(data)
    errors = 0
    for i in range(total):
        if data[i][0] != results[i][0]:
            errors += 1
    return float(errors/total)

assert(evaluate([['a','c','b'], ['a','a','b']], test_results) == 0.5)
assert(evaluate([['a','c','b'], ['b','a','b']], test_results) == 0)
assert(evaluate([['b','c','b'], ['a','a','b']], test_results) == 1)


# ## print_mean_variance
# * Takes in a list of error values
# * Returns the mean and variance of those error values

def print_mean_variance(errors):
    mean = float(sum(errors)/len(errors))
    print("Mean: {0}".format(mean))
    diffs = [(mean-error)**2 for error in errors]
    variance = float(sum(diffs)/(len(errors)-1))
    print("Variance: {0}\n".format(variance))
    return mean, variance #returned for testing purposes

'''
test_errors = [0.5, 1, 0]
test_mean, test_variance = print_mean_variance(test_errors)
assert(test_mean == 0.5)
assert(test_variance == 0.25)
test_errors = [1, 1, 1]
test_mean, test_variance = print_mean_variance(test_errors)
assert(test_variance == 0)
'''

# ## cross_validate
# * takes in a dataset as 2d numpy array, smoothing as a boolean, and folds as an int
# * Returns the mean and variance of the results

def cross_validate(data, smoothing, folds, test=False):
    att_vals = get_att_vals(data)
    np.random.shuffle(data)
    folds = np.array_split(data, folds)
    errors = []
    fold_num = 0
    for fold in folds:
        fold_num += 1
        split = int(0.8*(len(fold)))
        train_set = fold[:split]
        test_set = fold[split:]
        model = train(train_set, att_vals, smoothing)
        results = classify(model, test_set)
        error = float(evaluate(test_set, results))
        errors.append(error)
        print("Fold {0} error rate: {1}\n".format(fold_num, error))
    mean, variance = print_mean_variance(errors)
    if test: return mean, variance #returns for testing purposes only

'''
test_data=np.asarray([['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'],
['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'b']])
test_mean, test_variance = cross_validate(test_data, False, 2, True)
assert(test_mean == 0)
assert(test_variance == 0)
test_data=np.asarray([['a', 'b', 'a'], ['a', 'a', 'b'], ['b', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'a'], ['b', 'b', 'b'], ['a', 'b', 'b'],
['b', 'a', 'a'], ['b', 'b', 'a'], ['b', 'a', 'b'], ['a', 'a', 'a'], ['b', 'a', 'a'], ['a', 'b', 'b'], ['b', 'b', 'b']])
test_mean, test_variance = cross_validate(test_data, False, 2, True)
assert(test_mean != 0)
'''
