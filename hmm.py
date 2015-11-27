import numpy as np
import time
import theano
import random
import math

class Hmm:
    emsnprb = {}
    trnstnprb = []
    startprb = []
    endprb = []
    initprb =[]
    label = {}
    #label_ans = []

    label_map = {}
    label_keys = []
    label_num={}
    chr_map = {}
    chr_keys = []
    #dim_x = 0
    dim_y = 0
    @classmethod
    def loadgetprob(cls,label_map_file, chr_map_file, label_file, post_file,filename):
    	t_start = time.time()
        print 'load data...'
        cls.label_keys = np.genfromtxt(label_map_file, usecols=0, dtype=str)
        for i in xrange(len(cls.label_keys)): cls.label_num[cls.label_keys[i]] = i
        label_raw_data = np.genfromtxt(label_map_file, usecols=[1],dtype=str)
        for key, row in zip(cls.label_keys, label_raw_data):
            cls.label_map[key] = row

        cls.chr_keys = np.genfromtxt(chr_map_file, usecols=0, dtype=str)
        chr_raw_data = np.genfromtxt(chr_map_file, usecols=[1,2],dtype=None)
        for key, row in zip(cls.chr_keys, chr_raw_data):
            cls.chr_map[key] = row
        cls.dim_y = len(cls.label_keys)

        with open(label_file, 'r') as f:
            count = np.zeros(shape=(cls.dim_y, cls.dim_y),	dtype=float)
            cls.trnstnprb = np.zeros(shape=(cls.dim_y, cls.dim_y),	dtype=float)
            cls.startprb = np.zeros(shape=(cls.dim_y), dtype=float)
            cls.endprb = np.zeros(shape=(cls.dim_y), dtype=float)
            cls.initprb = np.zeros(shape=(cls.dim_y), dtype=float)
            countcol = np.zeros(shape=(cls.dim_y), dtype=float)
            frame = ''
            startcount = 0
            count_total =0;
            endcount =0
            postchar = -1
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                cls.label[token[0]] = token[1]
                #cls.label_ans.append(token[1])

                index = token[0].rfind('_')
                y = cls.label_num[token[1]]
                cls.initprb[y]+=1
                count_total+=1
                if not frame == token[0][0:index]:
                    if not postchar==-1:
                        cls.endprb[postchar]=+1
                        endcount += 1
                    cls.startprb[y] += 1
                    startcount += 1
                    postchar = y;
                    frame = token[0][0:index]
                else :					
                    count[postchar][y]+=1
                    countcol[postchar]+=1
                    postchar = y
            cls.endprb[postchar]=+1
            endcount += 1

            for i in xrange(cls.dim_y):
                cls.startprb[i] = cls.startprb[i]/startcount
                cls.endprb[i] = cls.endprb[i]/endcount
                cls.initprb[i] = cls.initprb[i]/count_total
                for j in xrange(cls.dim_y):
                    if(countcol[i]!=0): 
                        if(count[i][j]!=0):cls.trnstnprb[i][j] = count[i][j]/countcol[i] #i-->j

            cls.trnstnprb = np.log(cls.trnstnprb)
            cls.startprb = np.log(cls.startprb)
            cls.endprb = np.log(cls.endprb)
            cls.initprb = np.log(cls.initprb)
            #print cls.trnstnprb
        with open(filename,'r') as f:
            frame = ''
            temp_em = []
            for line in f:
                temp = []
                line = line.replace('\n','')
                token = line.split(' ')
                token[0] = token[0].replace(',','')
                index = token[0].rfind('_')
                #print token[0][0:index]
                if not frame == token[0][0:index]:
                    if not frame == '':
                        temp_em = np.log(temp_em)
                        cls.emsnprb[frame] = temp_em
                    temp_em = []
                    frame = token[0][0:index]
                for i in xrange(1, len(token)):
                	temp.append(float(token[i]))
                    #if(float(token[i])!=0):
                    #    temp.append(float(token[i]))
                temp_em.append(temp)
            cls.emsnprb[frame] = np.log(temp_em)
        t_end = time.time()
        print 'data loaded. loading time: %f sec' % (t_end - t_start)

    @classmethod
    def viterbi(cls):
    	t_start = time.time()
        print 'do viterbi...'
        mat = {}
        matTb = {}
        omxi = {}
        p_ans = {}
        path_ans = {}
        nrow = cls.dim_y
        for test_id in cls.emsnprb:
            ncol = len(cls.emsnprb[test_id])
            mat[test_id] = np.zeros(shape=(nrow,ncol), dtype=float) # prob
            matTb[test_id] = np.zeros(shape=(nrow,ncol), dtype=int) # backtrace	
            
            # Fill in first column
            for	i in xrange(nrow): # step1
                mat[test_id][i][0] = cls.initprb[i]+cls.emsnprb[test_id][0][i]+cls.startprb[i] # mat[phone][step]
            
            for j in xrange(1, ncol):
                for i in xrange(nrow):
                    ep = cls.emsnprb[test_id][j][i]/1.5 # emission probs
                    if ep == 0: continue
                    if j == ncol-1:
                        mx, mxi = mat[test_id][0][j-1] + cls.trnstnprb[0][i] + ep + cls.endprb[i], 0
                    else:
                        mx, mxi = mat[test_id][0][j-1] + cls.trnstnprb[0][i] + ep, 0

                    for i2 in xrange(1, nrow):
                        if j == ncol-1:
                            pr = mat[test_id][i2][j-1] + cls.trnstnprb[i2][i] + ep + cls.endprb[i]
                        else:
                            pr = mat[test_id][i2][j-1] + cls.trnstnprb[i2][i] + ep
                        if pr > mx:
                            mx, mxi = pr, i2
                    mat[test_id][i][j], matTb[test_id][i][j] = mx, mxi
            
            # Find final state with maximal probability
            omx, omxi[test_id] = mat[test_id][0][ncol-1], 0
            for i in xrange(1, nrow):
                if mat[test_id][i][ncol-1] > omx:
                    omx, omxi[test_id] = mat[test_id][i][ncol-1], i
            

            # Backtrace                        
            i,p,path = omxi[test_id],[omxi[test_id]],[cls.label_keys[omxi[test_id]]]
            #print omxi
            for j in xrange(ncol-1, 0, -1):
                i = matTb[test_id][i][j]
                p.append(i)
                path.append(cls.label_keys[i])
            p_ans[test_id] = p[::-1]
            path_ans[test_id] = path[::-1]
        t_end = time.time()
        print 'viterbi elapsed time: %f mins' % ( (t_end-t_start)/60.0 )
        return omx, p_ans, path_ans

    @classmethod
    def check(cls, fname):
        t_start = time.time()
        correct = 0
        count = 0
        with open(fname, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                if cls.label[token[0]] == token[1]: correct += 1
                count += 1
        t_end = time.time()
        print 'checking: %f sec' % (t_end - t_start)
        return correct, count

    @classmethod
    def check2(cls, path):
        t_start = time.time()
        correct = 0
        count = 0
        for i in path:
            for p in xrange(len(path[i])):
                frame = i + '_' + str(p+1)
                if cls.label[frame]== path[i][p]: correct += 1
                count += 1
        t_end = time.time()
        print 'checking: %f sec' % (t_end - t_start)
        return correct, count

    @classmethod
    def trim(cls, seq):
        # trim leading and tailing 'L' (sil)
        while (seq.startswith('L')):
            seq = seq[1:]
        while (seq.endswith('L')):
            seq = seq[:-1]
        
        # zip duplicate letters
        zipped = ''
        for i in xrange(len(seq)):
            if i==0:
                zipped = seq[i]
            elif seq[i] == zipped[-1]:
                continue
            else:
                zipped += seq[i]
        return zipped