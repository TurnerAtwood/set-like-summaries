import matplotlib.pyplot as plt
import numpy as np
import argv from sys
import pickle
import IR_Project

def main():
    if argv[1] == 'compute':
        for nr_sent in range(1,6):
            scores = IR_Project.run_test(nr_sent)
            pickle.dump(score,open(f'score_{nr_sent}','wb'))
            print(f'{nr_sent} sentences done...')
    else:
        rouge1 = []
        rougel = []
        for nr_sent in range(1,6):
            scores = pickle.load(open(f'score_{nr_sent}','rb'))
            score1 = 0
            scorel = 0
            for s in scores:
                score1 += s[0]['f']
                scorel += s[2]['f']
            score1 /= len(scores)
            scorel /= len(scores)
            rouge1.append(score1)
            rougel.append(scorel)
        fig,ax = plt.subplots()
        ax.plot([1,2,3,4,5],rouge1,label='Rouge-1')
        ax.plot([1,2,3,4,5],rougel,label='Rouge-l')
        ax.set_xlabel('Number of Sentences')
        ax.set_ylabel('f Score')
        ax.set_title('f Score vs. Sentences')
        ax.legend()
        ax.plot()

