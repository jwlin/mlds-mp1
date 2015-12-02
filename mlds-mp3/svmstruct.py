"""A module that SVM^python interacts with to do its evil bidding."""

# Thomas Finley, tfinley@gmail.com
import paths
from svmparser import TestParser
import numpy as np
import Levenshtein
import svmapi
import random

def parse_parameters(sparm):
    """Sets attributes of sparm based on command line arguments.
    
    This gives the user code a chance to change sparm based on the
    custom command line arguments.  The custom command line arguments
    are stored in sparm.argv as a list of strings.  The custom command
    lines are stored in '--option', then 'value' sequence.
    
    If this function is not implemented, any custom command line
    arguments are ignored and sparm remains unchanged."""
    sparm.arbitrary_parameter = 'I am an arbitrary parameter!'

def parse_parameters_classify(attribute, value):
    """Process a single custom command line argument for the classifier.

    This gives the user code a chance to change the state of the
    iclassifier based on a single custom command line argument, e.g.,
    one that begins with two dashes.  This function will be called
    multiple times if there are multiple custom command line
    arguments.

    If this function is not implemented, any custom command line
    arguments are ignored."""
    print 'Got a custom command line argument %s %s' % (attribute, value)

def read_examples(filename, sparm):
    print 'in example'
    if 'test' in filename:
        TestParser.load(paths.label_map_file, paths.chr_map_file, paths.mfcc_test, label_file=None)
    elif 'train' in filename:
        TestParser.load(paths.label_map_file, paths.chr_map_file, paths.mfcc_train, paths.label_file)
    else:
        print 'Exception in read_examples()'
    examples = []
    assert len(TestParser.x_seq) == len(TestParser.y_hat)
    for i in xrange(len(TestParser.x_seq)):
        #assert len(sparse_vec[i]) == len(TestParser.y_hat[i])
        tup = (TestParser.x_seq[i], TestParser.y_hat[i])
        examples.append(tup)
    print 'after example',len(examples)
    #return examples[0:5]
    return examples


def init_model(sample, sm, sparm):
    """Initializes the learning model.
    
    Initialize the structure model sm.  The sm.size_psi must be set to
    the number of features.  The ancillary purpose is to add any
    information to sm that is necessary from the user code
    perspective.  This function returns nothing."""
    # In our binary classification task, we've encoded a pattern as a
    # list of four features.  We just want a linear rule, so we have a
    # weight corresponding to each feature.  We also add one to allow
    # for a last "bias" feature.
    #sm.size_psi = len(sample[0][0])+1
    sm.size_psi = 39*48+48*48

def init_constraints(sample, sm, sparm):
    """Initializes special constraints.

    Returns a sequence of initial constraints.  Each constraint in the
    returned sequence is itself a sequence with two items (the
    intention is to be a tuple).  The first item of the tuple is a
    document object.  The second item is a number, indicating that the
    inner product of the feature vector of the document object with
    the linear weights must be greater than or equal to the number
    (or, in the nonlinear case, the evaluation of the kernel on the
    feature vector with the current model must be greater).  This
    initializes the optimization problem by allowing the introduction
    of special constraints.  Typically no special constraints are
    necessary.  A typical constraint may be to ensure that all feature
    weights are positive.

    Note that the slack id must be set.  The slack IDs 1 through
    len(sample) (or just 1 in the combined constraint option) are used
    by the training examples in the sample, so do not use these if you
    do not intend to share slack with the constraints inferred from
    the training data.

    The default behavior is equivalent to returning an empty list,
    i.e., no constraints."""
    import svmapi

    if True:
        # Just some example cosntraints.
        c, d = svmapi.Sparse, svmapi.Document
        # Return some really goofy constraints!  Normally, if the SVM
        # is allowed to converge normally, the second and fourth
        # features are 0 and -1 respectively for sufficiently high C.
        # Let's make them be greater than 1 and 0.2 respectively!!
        # Both forms of a feature vector (sparse and then full) are
        # shown.
        return [(d([c([(1,1)])],slackid=len(sample)+1),   1),
                (d([c([0,0,0,1])],slackid=len(sample)+1),.2)]
    # Encode positivity constraints.  Note that this constraint is
    # satisfied subject to slack constraints.
    constraints = []
    for i in xrange(sm.size_psi):
        # Create a sparse vector which selects out a single feature.
        sparse = svmapi.Sparse([(i,1)])
        # The left hand side of the inequality is a document.
        lhs = svmapi.Document([sparse], costfactor=1, slackid=i+1+len(sample))
        # Append the lhs and the rhs (in this case 0).
        constraints.append((lhs, 0))
    return constraints


def classify_example(x, sm, sparm):
    # W = [w_0, w_2, ..., w_47, wt_0_0, wt_0_1, wt_0_47, wt_1_0, wt_1_1, .., wt_1_47, .. wt_47_47]
    # w_i = w[i*39:(i+1)*39], i=0..47
    # wt_i_j = w[39*48 + i*48 + j], i=0..47, j=0..47
    print 'in class'
    w = sm.w
    #print len(x)
    #prob = [ [ (np.exp(np.dot(w[i*39:(i+1)*39],x[0])), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    prob = [ [ (np.log(1.0/48.0) + np.log(np.exp(np.dot(w[i*39:(i+1)*39],x[0]))), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    
    prev_state = []
    
    for m in xrange(1, len(x)):  # for frame x1, x2, .. x(m-1)
        # P(state s at iteraion m) = max { Prev_P(j)*P(s|j)*P(x_m|s), j=0..47}
        prob_m = []  # prob of each state at iteration m
        for s in xrange(48):
            #prob_s_m_all = [ (np.log(prob[m-1][i][0]) +np.log( np.exp(w[39*48+i*48+s])) +np.log( np.exp(np.dot(w[s*39:(s+1)*39],x[m]))) , i) for i in xrange(48) ]
            prob_s_m_all = [ (prob[m-1][i][0] + np.log(np.exp(w[39*48+i*48+s])) + np.log(np.exp(np.dot(w[s*39:(s+1)*39],x[m]))) , i) for i in xrange(48) ]
            max_prob = max(prob_s_m_all)[0]
            best_indices = [index for index in xrange(len(prob_s_m_all)) if prob_s_m_all[index][0] == max_prob]
            chosen_index = random.choice(best_indices)  # Pick randomly among the best
            prob_m.append(prob_s_m_all[chosen_index])
        prob.append(prob_m)
        
    # backtrack based on prob[m-1]
    ybar=[]
    max_state = max(prob[len(x)-1])[1]
    ybar.append(max_state)
    for m in xrange(len(x)-2,-1,-1):
        max_state= prob[m][max_state][1]
        ybar.append(max_state)
    return ybar[::-1]

def find_most_violated_constraint(x, y, sm, sparm):
    # W = [w_0, w_2, ..., w_47, wt_0_0, wt_0_1, wt_0_47, wt_1_0, wt_1_1, .., wt_1_47, .. wt_47_47]
    # w_i = w[i*39:(i+1)*39], i=0..47
    # wt_i_j = w[39*48 + i*48 + j], i=0..47, j=0..47
    print 'in find'
    w = sm.w

    prob = [ [ (np.log(1.0/48.0) + np.log(np.exp(np.dot(w[i*39:(i+1)*39],x[0]))), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    #print 'initial prob in step 0', prob
    prev_state = []
    
    for m in xrange(1, len(x)):  # for frame x1, x2, .. x(m-1)
        # P(state s at iteraion m) = max { Prev_P(j)*P(s|j)*P(x_m|s), j=0..47}
        prob_m = []  # prob of each state at iteration m
        for s in xrange(48):
            #print m ,s
            #print 'prob[m-1][i][0]', prob[m-1][i][0] 
            #print 'np.log( np.exp(w[39*48+i*48+s]))', np.log( np.exp(w[39*48+i*48+s]))
            #print 'np.log( np.exp(np.dot(w[s*39:(s+1)*39],x[m])))', np.log( np.exp(np.dot(w[s*39:(s+1)*39],x[m])))
            prob_s_m_all = [ (prob[m-1][i][0] + np.log(np.exp(w[39*48+i*48+s])) + np.log(np.exp(np.dot(w[s*39:(s+1)*39],x[m]))) , i) for i in xrange(48) ]
            #print 'prob for state', s, 'in step', m, ':', prob_s_m_all
            max_prob = max(prob_s_m_all)[0]
            best_indices = [index for index in xrange(len(prob_s_m_all)) if prob_s_m_all[index][0] == max_prob]
            chosen_index = random.choice(best_indices)  # Pick randomly among the best
            #print 'max prob for state', s, 'in step', m, ':', prob_s_m_all[chosen_index]
            prob_m.append(prob_s_m_all[chosen_index])
        #print prob
        prob.append(prob_m)
        
    # backtrack based on prob[m-1]
    ybar=[]
    max_state = max(prob[len(x)-1])[1]
    ybar.append(max_state)
    for m in xrange(len(x)-2,-1,-1):
        max_state= prob[m][max_state][1]
        ybar.append(max_state)
    return ybar[::-1]


def find_most_violated_constraint_slack(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for slack rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for margin rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def psi(x, y, sm, sparm):
    print 'in psi'
    prev_num = -1
    feature_vector = [0]*(39*48+48*48)
    for i in xrange(len(x)):
        offset = y[i]*39
        for j in xrange(39):
            feature_vector[offset+j] += x[i][j]
        base = 39*48
        if prev_num >= 0:
            offset = prev_num*48
            feature_vector[base + offset + y[i]] = 1  # or +=1?
        prev_num = y[i]
    
    # do sparse format
    sparse_vec = []
    for i in xrange(len(feature_vector)):
        if feature_vector[i] != 0:
            sparse_vec.append((i+1, feature_vector[i]))
    sparse_vec = svmapi.Sparse(sparse_vec)
    return sparse_vec

def loss(y, ybar, sparm):
    print 'in loss'
    y_c = []
    ybar_c = []
    for i in xrange(len(y)):
        y_c.append(chr(y[i]+64)) 
        ybar_c.append(chr(ybar[i]+64))
    print 'y:', ''.join(y_c)
    print 'ybar:', ''.join(ybar_c)
    d = Levenshtein.distance(''.join(y_c), ''.join(ybar_c))
    print 'd:', d
    return d

def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called just before the end of each cutting plane iteration.

    This is called just before the end of each cutting plane
    iteration, primarily to print statistics.  The 'ceps' argument is
    how much the most violated constraint was violated by.  The
    'cached_constraint' argument is true if this constraint was
    constructed from the cache.
    
    The default behavior is that nothing is printed."""
    print

def print_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    The default behavior is that nothing is printed."""
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
    print 'Losses:',
    print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]

def print_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.
    
    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    The default behavior is that nothing is printed."""
    print teststats

def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    if exnum==0: teststats = []
    print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.
    
    Write the structmodel sm to a file at path filename.

    The default behavior is equivalent to
    'cPickle.dump(sm,bz2.BZ2File(filename,'w'))'."""
    import cPickle, bz2
    f = bz2.BZ2File(filename, 'w')
    cPickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    """Load the structure model from a file.
    
    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    The default behavior is equivalent to
    'return cPickle.load(bz2.BZ2File(filename))'."""
    import cPickle, bz2
    return cPickle.load(bz2.BZ2File(filename))

def write_label(fileptr, y):
    """Write a predicted label to an open file.

    Called during classification, this function is called for every
    example in the input test file.  In the default behavior, the
    label is written to the already open fileptr.  (Note that this
    object is a file, not a string.  Attempts to close the file are
    ignored.)  The default behavior is equivalent to
    'print>>fileptr,y'"""
    print>>fileptr,y

def print_help():
    """Help printed for badly formed CL-arguments when learning.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    import svmapi
    print svmapi.default_help
    print "This is a help string for the learner!"

def print_help_classify():
    """Help printed for badly formed CL-arguments when classifying.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    print "This is a help string for the classifer!"
