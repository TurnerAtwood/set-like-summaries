import matplotlib.pyplot as plt
import numpy as np
from sys import argv 
import pickle
import IR_Project

def main():
    sim_funcs = {IR_Project.cosine:'Cosine'}
    #sim_funcs = {IR_Project.cosine:'Cosine', IR_Project.euclidean:'Euclidean', IR_Project.manhattan:'Manhattan'}
    redundancy = [True, False]
    #redundancy = [False]
    max_sent = 6

    if len(argv) == 2 and argv[1] == 'compute':
        all_topic = IR_Project.read_data()
        topic_data = []
        for topic in all_topic:
            if topic[1]:
                topic_data.append(topic)
        print(f'{len(topic_data)} Topics with Summaries')

        scores = {}
        for red in redundancy:
            for sim in sim_funcs:
                for nr_sent in range(1, max_sent):
                    scores[(red,sim,nr_sent)] = IR_Project.calc_scores(topic_data, nr_sent, sim, red)
                    print(f'{nr_sent} sentence summaries done...')
                print('Similarity function done...')
            print('Redundant included done...' if red else 'Redundant excluded done...')

        pickle.dump(scores,open(f'scores','wb'))
    scores = pickle.load(open(f'scores','rb'))
    for sim in sim_funcs:
        fig,ax = plt.subplots()
        for red in redundancy:
            rouge1 = []
            rougel = []
            for nr_sent in range(1, max_sent):
                score1 = 0
                scorel = 0
                tot_points = len(scores[(red,sim,nr_sent)])
                for s in scores[(red,sim,nr_sent)]:
                    for datapoint in s:
                        score1 += datapoint[0]['rouge-1']['f']
                        scorel += datapoint[0]['rouge-l']['f']
                score1 /= tot_points * 2
                scorel /= tot_points * 2
                rouge1.append(score1)
                rougel.append(scorel)
            label_prefix = 'Redundant ' if red else 'Non-Redundant '
            ax.plot(list(range(1,max_sent)),rouge1,label=label_prefix + 'Rouge-1')
            ax.plot(list(range(1,max_sent)),rougel,label=label_prefix + 'Rouge-l')
        ax.set_xlabel('Number of Sentences')
        ax.set_ylabel('f Score')
        ax.set_title(f'f Score vs. Sentences: {sim_funcs[sim]}')
        ax.legend()
        #ax.plot()
        plt.show()

if __name__ == "__main__":
    main()
