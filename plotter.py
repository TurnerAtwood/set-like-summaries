from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import pickle
import project

def main():
    #sim_funcs = {project.cosine:'Cosine', project.euclidean:'Euclidean', project.manhattan:'Manhattan', project.random_selection:'Random'}
    #redundancy = [True, False]
    #score_type = ['1','2','l']
    # Redundant vs NonRedundant
    sim_funcs = {project.cosine:'Cosine', project.random_selection:'Random'}
    redundancy = [True, False]
    title = f'f Score vs. Sentences: Redundancy'
    score_type = ['1','2','l']
    mult = False
    # 1gram all 4
    sim_funcs = {project.cosine:'Cosine', project.euclidean:'Euclidean', project.manhattan:'Manhattan', project.random_selection:'Random'}
    redundancy = [False]
    title = 'f Score vs. Sentences: Rouge-'
    score_type = ['1','2','1']
    mult = True
    max_sent = 6

    if len(argv) == 2 and argv[1] == 'compute':
        all_topic = project.read_data()
        topic_data = []
        for topic in all_topic:
            if topic[1]:
                topic_data.append(topic)
        print(f'{len(topic_data)} Topics with Summaries')

        scores = {}
        for red in redundancy:
            for sim in sim_funcs:
                for nr_sent in range(1, max_sent):
                    scores[(red,sim,nr_sent)] = project.calc_scores(topic_data, nr_sent, sim, red)
                    print(f'{nr_sent} sentence summaries done...')
                print('Similarity function done...')
            print('Redundant included done...' if red else 'Redundant excluded done...')
        
        pickle.dump(scores,open(f'scores','wb'))
    scores = pickle.load(open(f'scores','rb'))
    if mult:
        for sc in score_type:
            for sim in sim_funcs:
                fig,ax = plt.subplots()
                for red in redundancy:
                    rouges = {'1':[], '2':[], 'l':[]}
                    for nr_sent in range(1, max_sent):
                        score1 = 0
                        scorel = 0
                        score2 = 0
                        tot_points = len(scores[(red,sim,nr_sent)])
                        for s in scores[(red,sim,nr_sent)]:
                            for datapoint in s:
                                score1 += datapoint['rouge-1']['f']
                                scorel += datapoint['rouge-l']['f']
                                score2 += datapoint['rouge-2']['f']
                        score1 /= tot_points * 2
                        scorel /= tot_points * 2
                        score2 /= tot_points * 2
                        rouges['1'].append(score1)
                        rouges['l'].append(scorel)
                        rouges['2'].append(score2)
                    label_prefix = ('Redundant ' if red else 'Non-Redundant ') if  len(redundancy) > 1 else ''
                    for rouge in score_type:
                        ax.plot(list(range(1,max_sent)),rouges[rouge],label=label_prefix + f'Rouge-[rouge]')
            ax.set_xlabel('Number of Sentences')
            ax.set_ylabel('f Score')
            ax.set_title(title+sc)
            ax.legend()
            #ax.plot()
            plt.show()
    else:
        for sim in sim_funcs:
            fig,ax = plt.subplots()
            for red in redundancy:
                rouges = {'1':[], '2':[], 'l':[]}
                for nr_sent in range(1, max_sent):
                    score1 = 0
                    scorel = 0
                    tot_points = len(scores[(red,sim,nr_sent)])
                    for s in scores[(red,sim,nr_sent)]:
                        for datapoint in s:
                            score1 += datapoint[0]['rouge-1']['f']
                            scorel += datapoint[0]['rouge-l']['f']
                            score2 += datapoint[0]['rouge-2']['f']
                    score1 /= tot_points * 2
                    scorel /= tot_points * 2
                    score2 /= tot_points * 2
                    rouges['1'].append(score1)
                    rouges['l'].append(scorel)
                    rouges['2'].append(score2)
                label_prefix = ('Redundant ' if red else 'Non-Redundant ') if  len(redundancy) > 1 else ''
                for rouge in score_type:
                    ax.plot(list(range(1,max_sent)),rouges[rouge],label=label_prefix + f'Rouge-[rouge]')
        ax.set_xlabel('Number of Sentences')
        ax.set_ylabel('f Score')
        ax.set_title(title)
        ax.legend()
        #ax.plot()
        plt.show()
    

if __name__ == "__main__":
    main()
