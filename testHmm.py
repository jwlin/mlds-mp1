from hmm import Hmm
import time

post_file = '../dataset/posteriorgram/test3.post'#'../dataset/posteriorgram/train2.post'
test_file = '../dataset/posteriorgram/test3.post'
label_file = '../dataset/label/train.lab'
label_map_file = '../dataset/phones/48_39.map'
chr_map_file = '../dataset/48_idx_chr.map_b'
sol_train_file = 'sol_train.txt'

Hmm.loadgetprob(label_map_file, chr_map_file, label_file, post_file, post_file)#'test_y_.txt'
n, p, path = Hmm.viterbi()

'''with open(sol_train_file,'w') as f:
    #f.write('Id,Prediction\n')
    for i in path:
        for p in xrange(len(path[i])):
            f.write( i + '_' + str(p+1) + ',' + str(path[i][p]) + '\n')

# not to compare w/ file
count,counttotal = Hmm.check(sol_train_file)
print (str(count) +'/'+ str(counttotal))'''


#count,counttotal = Hmm.check2(path)
#print (str(count) +'/'+ str(counttotal))
hw1_sol = {}
with open('solution_hw1.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in path:
        hw1_sol[i] = []
        for p in xrange(len(path[i])):
            frame = i + '_' + str(p+1)
            label = Hmm.label_map[str(path[i][p])]
            hw1_sol[i].append(label)
            f.write( frame + ',' + label + '\n')

with open('solution_hw2.csv', 'w') as f:
    f.write('id,phone_sequence\n')
    for frame in hw1_sol:
        tmp_seq=''
        for j in xrange(len(hw1_sol[frame])):
            tmp_seq += str(Hmm.chr_map[ hw1_sol[frame][j] ][1])
        tmp_seq = Hmm.trim(tmp_seq)
        f.write(frame + ',' + tmp_seq + '\n')
print('done')

