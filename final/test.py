import nltk
from gensim.models import Word2Vec 
from gensim.models import word2vec 
import numpy as np

choices_answer = ['A','B','C','D','E']

stopwords = []
for l in open('stopwords.txt'):
    stopwords.append(l.rstrip('\n'))

def content_fraction(text):
    '''stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('doing')
    stopwords.append(u',')
    stopwords.append(u'?')
    stopwords.append(u'[')
    stopwords.append(u']')
    stopwords.append(u'(')
    stopwords.append(u')')
    stopwords.append(u'"')
    stopwords.append(u'\'')'''
    
    #f = open('stopwords.txt','r')
  
    return [w for w in text if w.lower() not in stopwords]

    
def answer_spilt(ans):
    
    a = ans.split('(B)')[0].split('(A)')[1].rstrip('  ')
    b = ans.split('(B)')[1].split('(C)')[0].rstrip('  ')
    c = ans.split('(B)')[1].split('(C)')[1].split('(D)')[0].rstrip('  ')
    d = ans.split('(B)')[1].split('(C)')[1].split('(D)')[1].split('(E)')[0].rstrip('  ')
    e = ans.split('(B)')[1].split('(C)')[1].split('(D)')[1].split('(E)')[1].rstrip('  ')
    return [a,b,c,d,e]

    
def load_data(q,cho,ans,ann):
    data = {}
    
    for line in open(q):
        l =  line.rstrip('\n').split(None,2)
        data.setdefault(l[1], [])
        data[l[1]].append(l[0])
        data[l[1]].append(l[2])
        #print data[q[1]]
        
    for line in open(cho):
        l =  line.rstrip('\n').split(None,2)
        data[l[1]].append(l[2])
    
    for line in open(ans):
        l =  line.rstrip('\n').split(None,2)
        data[l[1]].append(l[2])
    
        
    for line in open(ann):
        l =  line.rstrip('\n').split(None,2)
        data[l[1]].append(l[2])
    
        
    return data
    
correct = 0
total = 0    
data = load_data('final_project_pack/question.train','final_project_pack/choices.train','final_project_pack/answer.train_sol','final_project_pack/annotation.train')
for key in data.keys():
    #print data[key]
    #data = raw_input(">>> question: ")
    question = unicode(data[key][1])

    text = nltk.word_tokenize(question)

    #print nltk.pos_tag(text)
    text = content_fraction(text)

    #answer = raw_input(">>> ans: ")
    answer = answer_spilt(data[key][2])
    #sentences = word2vec.Text8Corpus('enwik9/enwik9')
    #model = word2vec.Word2Vec(sentences, size=200)
    model = Word2Vec.load('text8/text8.model')
    #print text
    #print answer
    n_similarity_list =[]
    for i in answer:
        maxvalue = -1
        '''for j in text:
            try:
                maxvalue +=  model.similarity(i,j)
                print model.similarity(i,j) , i ,j
                
            except KeyError:
                pass'''
        try:
            temp =[]
            temp.append(i)
            maxvalue = model.n_similarity(text,temp)
        except KeyError:
                pass
        n_similarity_list.append(maxvalue)
       
    similarity_list =[]
    for i in answer:
        maxvalue = 0
        for j in text:
            try:
                maxvalue +=  model.similarity(i,j)
                #print model.similarity(i,j) , i ,j
                
            except KeyError:
                maxvalue += -1
        similarity_list.append(maxvalue)
    for i in xrange(len(similarity_list)):
        similarity_list[i]+=n_similarity_list[i]
        
    if  choices_answer[similarity_list.index(max(similarity_list))] == data[key][3]:
        correct+=1
    total+=1
    print   data[key][1]
    print   data[key][2]
    print   'choice = ',choices_answer[similarity_list.index(max(similarity_list))] ,'ans =', data[key][3]
    print   correct/float(total)
