import numpy

result_file = 'Result.rlt'
file_48_39 = 'data/phones/48_39.map'
file_48_idx_chr = 'data/48_idx_chr.map_b'
test_file = 'data/mfcc/test.ark'

ph48 = numpy.loadtxt(file_48_39, dtype='str_',usecols=(0,))
ph48_39 = numpy.loadtxt(file_48_39, dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)

letter = numpy.loadtxt(file_48_idx_chr, dtype='str_',usecols=(0,))
ans = numpy.loadtxt(file_48_idx_chr, dtype='str_',usecols=(2,))
ans_dict = {}
for l, a in zip(letter, ans):
    ans_dict[l] = a

frames = numpy.loadtxt(test_file, dtype='str_', usecols=(0,))
utterances = []
prev_utter = ''
for f in frames:
    utter_spliter = f.rfind('_')
    utter = f[0:utter_spliter]
    if not prev_utter == utter:
        utterances.append(utter)
        prev_utter = utter

solution_1 = []  # for solution1.csv, letters for each frame
solution_3 = []  # for solution3.csv, setences for each utterance
with open (result_file) as f:
    for line in f:
        line = line.replace('\n', '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        frame_ids = line.split(',')
        frame_letters = []
        for i in xrange(len(frame_ids)):
            frame_letters.append(ph48[int(frame_ids[i])])
        #print frame_letters
        for i in xrange(len(frame_letters)):
            frame_letters[i] = ph48_39_dict[frame_letters[i]]
        solution_1 += frame_letters
        #print frame_letters
        ans_letters = list(frame_letters)
        for i in xrange(len(ans_letters)):
            ans_letters[i] = ans_dict[ans_letters[i]]
        #print frame_letters
        seq = ''.join(ans_letters)

        # trim leading and tailing 'L' (sil)
        while (seq.startswith('L')):
            seq = seq[1:]
        while (seq.endswith('L')):
            seq = seq[:-1]
        
        
        
        for i in xrange(1, len(seq)-1):
            if seq[i]!=seq[i+1] and seq[i]!=seq[i-1]:
                seq[i]=seq[i-1]
        for i in xrange(1, len(seq)-2):
            if seq[i] == seq [i+1]:
                if seq[i] != seq[i-1] and seq[i] != seq[i+2]:
                    seq[i] = seq[i-1]
                    seq[i+1] = seq[i-1]
            elif seq[i] != seq[i+1]:
                pass
                
                
        # zip duplicate letters
        zipped = ''
        for i in xrange(len(seq)):
            if i==0:
                zipped = seq[i]
            elif seq[i] == zipped[-1]:
                continue
            else:
                zipped += seq[i]
        solution_3.append(zipped)

print len(frames), len(solution_1)
print len(utterances), len(solution_3)
assert len(frames) == len(solution_1)
with open('solution1.csv', 'a') as s1:
    s1.write('Id,Prediction\n')
for i in xrange(len(frames)):
    with open('solution1.csv', 'a') as s1:
        s1.write(frames[i] + ',' + solution_1[i] + '\n')

assert len(utterances) == len(solution_3)
with open('solution3.csv', 'a') as s3:
    s3.write('id,phone_sequence\n')
for i in xrange(len(utterances)):
    with open('solution3.csv', 'a') as s3:
        s3.write(utterances[i] + ',' + solution_3[i] + '\n')

