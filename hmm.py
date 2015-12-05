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
    def loadgetprob(cls,label_map_file, chr_map_file, label_file, post_file):
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
            count = np.zeros(shape=(cls.dim_y, cls.dim_y), dtype=float)
            cls.trnstnprb = np.zeros(shape=(cls.dim_y, cls.dim_y), dtype=float)
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

        with open(post_file,'r') as f:
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
                temp_em.append(temp)
            cls.emsnprb[frame] = np.log(temp_em)
        
        cls.trnstnprb = np.log(cls.trnstnprb)
        cls.startprb = np.log(cls.startprb)
        cls.endprb = np.log(cls.endprb)
        cls.initprb = np.log(cls.initprb)
        t_end = time.time()
        print 'data loaded. loading time: %f sec' % (t_end - t_start)

    # how to use this function??????(what are the input of X&Y)
    @classmethod    
    def add_log(cls, X, Y):
        if(Y>X):
            temp = X;
            X=Y;
            Y=temp;

        diff = Y-X
        if(diff < -100.0): return X;
        else: return X+np.log(np.exp(diff)+1.0);

    @classmethod
    def viterbi(cls, setting):
    	t_start = time.time()
        print 'do viterbi...'
        mat = {}
        matTb = {}
        omxi = {}
        p_ans = {}
        path_ans = {}
        nrow = cls.dim_y
        cls.trnstnprb /= setting
        for test_id in cls.emsnprb:
            ncol = len(cls.emsnprb[test_id])
            mat[test_id] = np.zeros(shape=(nrow,ncol), dtype=float) # prob
            matTb[test_id] = np.zeros(shape=(nrow,ncol), dtype=int) # backtrace	
            
            # Fill in first column
            for	i in xrange(nrow): # step1
                #mat[test_id][i][0] = Hmm.add_log(cls.initprb[i], cls.emsnprb[test_id][0][i])
                #mat[test_id][i][0] = Hmm.add_log(mat[test_id][i][0], cls.startprb[i])
                mat[test_id][i][0] = cls.initprb[i]+cls.emsnprb[test_id][0][i]+cls.startprb[i] # mat[phone][step]
            
            for j in xrange(1, ncol):
                for i in xrange(nrow):
                    ep = cls.emsnprb[test_id][j][i]# emission probs
                    if ep == 0: continue
                    '''
                    mxi=0
                    mx = Hmm.add_log(mat[test_id][0][j-1], cls.trnstnprb[0][i])
                    mx = Hmm.add_log(mx, ep)
                    if j == ncol-1: mx = Hmm.add_log(mx, cls.endprb[i])'''
                    if j == ncol-1:
                        mx, mxi = mat[test_id][0][j-1] + cls.trnstnprb[0][i] + ep + cls.endprb[i], 0
                    else:
                        mx, mxi = mat[test_id][0][j-1] + cls.trnstnprb[0][i] + ep, 0

                    for i2 in xrange(1, nrow):
                        '''
                        pr = Hmm.add_log(mat[test_id][i2][j-1], cls.trnstnprb[i2][i])
                        pr = Hmm.add_log(pr, ep)
                        if j == ncol-1: pr = Hmm.add_log(pr, cls.endprb[i])'''
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
    def check(cls, path):
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

    @classmethod
    def save_result(cls, path, setting):
        t_start = time.time()
        hw1_sol={}
        with open('solution_hw1_'+str(setting)+'.csv', 'w') as f:
            f.write('Id,Prediction\n')
            for i in path:
                hw1_sol[i] = []
                for p in xrange(len(path[i])):
                    frame = i + '_' + str(p+1)
                    label = Hmm.label_map[str(path[i][p])]
                    hw1_sol[i].append(label)
                    f.write( frame + ',' + label + '\n')

        with open('solution_hw2_'+str(setting)+'.csv', 'w') as f:
            f.write('id,phone_sequence\n')
            for frame in hw1_sol:
                tmp_seq=''
                for j in xrange(len(hw1_sol[frame])):
                    tmp_seq += str(Hmm.chr_map[ hw1_sol[frame][j] ][1])
                tmp_seq = Hmm.trim(tmp_seq)
                f.write(frame + ',' + tmp_seq + '\n')
        
        # smooth
        '''for i in hw1_sol:
            hw1_sol[i][0]='sil'
            hw1_sol[i][1]='sil'
            hw1_sol[i][len(hw1_sol[i])-1]='sil'
            hw1_sol[i][len(hw1_sol[i])-2]='sil'
            last = 'sil'
            for j in xrange(2, len(hw1_sol[i])-2):
                cur = hw1_sol[i][j]
                if cur!=last:
                    if cur==hw1_sol[i][j+1]:
                        last = cur
                    else:
                        hw1_sol[i][j]=last'''
        '''for i in hw1_sol:
            hw1_sol[i][0]='sil'
            hw1_sol[i][1]='sil'
            hw1_sol[i][2]='sil'
            hw1_sol[i][len(hw1_sol[i])-1]='sil'
            hw1_sol[i][len(hw1_sol[i])-2]='sil'
            hw1_sol[i][len(hw1_sol[i])-3]='sil'
            last = 'sil'
            for j in xrange(3, len(hw1_sol[i])-3):
                cur = hw1_sol[i][j]
                if cur!=last:
                    if cur==hw1_sol[i][j+1] and cur==hw1_sol[i][j+2]:
                        last = cur
                    else:
                        hw1_sol[i][j]=last

        with open('solution_hw1_smooth_'+str(setting)+'.csv', 'w') as f:
            f.write('Id,Prediction\n')
            for i in hw1_sol:
                for p in xrange(len(hw1_sol[i])):
                    frame = i + '_' + str(p+1)
                    f.write( frame + ',' + hw1_sol[i][p] + '\n')

        with open('solution_hw2_smooth_'+str(setting)+'.csv', 'w') as f:
            f.write('id,phone_sequence\n')
            for frame in hw1_sol:
                tmp_seq=''
                for j in xrange(len(hw1_sol[frame])):
                    tmp_seq += Hmm.chr_map[ hw1_sol[frame][j] ][1]
                tmp_seq = Hmm.trim(tmp_seq)
                f.write(frame + ',' + tmp_seq + '\n')'''
        t_end = time.time()
        print 'result save: %f sec' % (t_end - t_start)