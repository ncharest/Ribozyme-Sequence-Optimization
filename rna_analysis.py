# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:54:44 2020

@author: Nate
"""
    
    #%% Initialization
    
import pickle
import matplotlib.pyplot as plt
import itertools as itt
import numpy as np
import copy
import networkx as nx
import time
import data_structures as DS
from sklearn import svm
from sklearn import preprocessing
import random
import json
import data_structures as DS
#%% Helper Functions ##########

def let2vec(let):
    value = np.asarray([0]*4)
    for i in let:
        if i == 'A':
            value += np.asarray([1,0,0,0])
        if i == 'C': 
            value += np.asarray([0,1,0,0])
        if i == 'T':
            value += np.asarray([0,0,1,0])
        if i == 'G':
            value += np.asarray([0,0,0,1])
    return value

def list_to_string(list_seq):
    output = ''
    for i in list_seq:
        output += i
    return output

def grain(value, floor, roof, bins):
    stride = (roof - floor)/bins
    bins = np.arange(floor, roof+stride, stride)
    class_counter = -int(len(bins)/2)
    if value < bins[0]:
        return (class_counter, bins)
    for i in range(len(bins)):
        class_counter += 1
        try:
            if value > bins[i] and value<bins[i+1]:
                return (class_counter, bins)
        except:
            return (len(bins)+-int(len(bins)/2), bins)
        



### Functions for Class seq
def compare_sequences(seq1, seq2):
    order = 0
    mutations = []
    locations = []
    for i in range(len(seq1.sequence)):
        if seq1.sequence[i] != seq2.sequence[i]:
            order += 1
            mutations.append(str(seq1.sequence[i])+str(i)+str(seq2.sequence[i]))
            locations.append(i)
    return order, mutations, locations

def mutation_code(first, second, resnum, total_res):
    molecules = 'ACTG'
    vector = np.zeros(len(list(itt.permutations(molecules, 2)))*total_res)
    for i in range(len(list(itt.permutations(molecules, 2)))):
        if (first, second) == list(itt.permutations(molecules, 2))[i]:
            mut_type = i
    res_id = mut_type+resnum*len(list(itt.permutations(molecules, 2)))
    vector[res_id] = 1
    return vector

def encode_sequence(seq):
    encoded_rep = []
    vector_rep = []
    for i in seq:
        if i == 'A':
            encoded_rep.append([1,0,0,0])
            vector_rep.append(1)
            vector_rep.append(0)
            vector_rep.append(0)
            vector_rep.append(0)
        elif i == 'C':
            encoded_rep.append([0,1,0,0])
            vector_rep.append(0)
            vector_rep.append(1)
            vector_rep.append(0)
            vector_rep.append(0)
        elif i == 'T':
            encoded_rep.append([0,0,1,0])
            vector_rep.append(0)
            vector_rep.append(0)
            vector_rep.append(1)
            vector_rep.append(0)
        elif i == 'G':
            encoded_rep.append([0,0,0,1])
            vector_rep.append(0)
            vector_rep.append(0)
            vector_rep.append(0)
            vector_rep.append(1)
    return np.asarray(encoded_rep), np.asarray(vector_rep)

######### Class Definitions

class seq:
    def __init__(self, data, index = 'NA', stats_value = 'N/A'):
        self.atts = {}
        self.index = index
        self.sequence = data['seq']
        self.list_seq = list(self.sequence)
        self.seq_len = len(self.sequence)
        self.matrix_rep, self.vector_rep = encode_sequence(self.sequence)
        
        try:
            self.atts.update({'k' : data['pointEstimation']['k']})
            self.atts.update({'kA' : data['pointEstimation']['kA']})
        except:
            pass
    
    def assign_class(self, class_threshold = 7.0):
        if self.atts['kA'] > class_threshold:
            self.act = 1
        elif self.atts['kA'] < class_threshold:
            self.act = 0
        else:
            print("Didn't assign class "+str(self.sequence)+str(self.atts['kA']))
        
    def embed(self, embedding_length):
        self.embedding = []
        for i in range(len(self.sequence)-(embedding_length - 1)):
            self.embedding.append(tuple(self.sequence[i:i+embedding_length]))
        self.archetypes = list(itt.product('ACTG', repeat = embedding_length))
        self.arch_counts = []
        self.total = 0
        self.PE = 0.0
        for i in self.archetypes:
            self.arch_counts.append([i, 0])
            for j in self.embedding:
                self.total += 1
                if j == i:
                    self.arch_counts[-1][1] += 1
        for i in self.arch_counts:
            ### compute KL divergence between sequence representation and uniform distribution of permutations
            if i[1] != 0:
                self.PE += -(i[1] / self.total) * np.log((i[1] / self.total)/(1/len(self.archetypes)))
                
    def calc_metrics(self):
        self.percent_A = 0.0
        self.percent_G = 0.0
        self.percent_T = 0.0
        self.percent_C = 0.0
        for i in self.sequence:
            if i == 'A':
                self.percent_A += 1
            if i == 'G':
                self.percent_G += 1
            if i == 'T':
                self.percent_T += 1
            if i == 'C':
                self.percent_C += 1
        self.percent_A /= self.seq_len
        self.percent_C /= self.seq_len
        self.percent_G /= self.seq_len
        self.percent_T /= self.seq_len
        self.symmetry = 0
        self.pair_dic = {'A' : 'T', 'T' : 'A', 'C' : 'G', 'G' : 'C'}
        for i in range(int(len(self.sequence)/2)):
            if self.sequence[i] == self.sequence[-(i+1)]:
                self.symmetry += 1
        self.pair_symmetry = 0
        for i in range(int(len(self.sequence)/2)):
            if self.sequence[i] == self.pair_dic[self.sequence[-(i+1)]]:
                self.pair_symmetry += 1
        
    def gen_nearest_neighbors(self):
        self.nearest_neighbors = []
        for i in range(self.seq_len):
            self.test_seq = list(copy.deepcopy(self.sequence))
            for j in self.alphabet:
                self.test_seq[i] = j
                if self.test_seq != self.list_seq:
                    self.nearest_neighbors.append(list_to_string(self.test_seq))
    
    def learn_library(self, lib):
        self.lib = lib
        
    def generate_locality(self, distance):
        self.local_group = []
        for i in self.lib.data:
            if compare_sequences(self, j)[0] <= int(distance):
                self.local_group.append(j.index)
        
    def detect_neighbors(self, lib, sorting = 'N'):
        self.neighbors = []
        self.orders = len(self.sequence) - np.tensordot(self.lib.matrix_rep, self.matrix_rep)
        for j in range(len(self.orders)):
            if self.orders[j] == 1:
                self.neighbors.append(self.lib.data[j])
        if sorting != 'N':
            self.neighbors.sort(key = lambda X : X.atts['kA'], reverse=True)
                    
    def characterize_mutations(self):
        self.neigh_mut = []
        for i in self.neighbors:
            self.neigh_mut.append(mutation(self, i))
            self.lib.add_mutation(mutation(self, i))
                       
class mutation:
    def __init__(self, WT, M, stats_lib = 'n/a'):
        self.WT = WT
        self.M = M
        self.seqWT = WT.sequence
        self.seqM = M.sequence
        self.vectors = []
        self.stats_lib = stats_lib

        for i in range(len(self.seqWT)):
            if self.seqWT[i] != self.seqM[i]:
                self.abv = self.seqWT[i]+str(i)+self.seqM[i]
                self.site = i
                self.vectors.append(mutation_code(self.seqWT[i], self.seqM[i], i, len(self.seqWT)))
        self.deltas = {}
        self.deltas.update({'kA' : (WT.atts['kA'] - M.atts['kA'])})
        try:
            self.pvalue = stats_lib[WT.sequence][M.sequence]
        except:
            try: 
                self.pvalue = stats_lib[M.sequence][WT.sequence]
            except:
                pass
        
    def reverse(self):
        return mutation(self.M, self.WT, self.stats_lib)
        
    def learn_library(self, lib):
        self.lib = lib   
        
      
class pathway:
    def __init__(self, seq1, seq2):
        self.order, self.formula = compare_sequences(seq1, seq2)  
        self.dummy_strand = list(seq1.sequence)
        self.types = [ (i[0], i[-1], i[1:-1]) for i in self.formula ]
        self.thread = [ (i[0], i[-1]) for i in self.formula]
        self.prior_test_samples = ['']
        self.test_samples = ['']
        self.path_sequences = []
        self.position_dict = {}
        for i in range(len(self.types)):
            self.position_dict.update({self.types[i][-1] : i})
    def create_pathway(self):
        for i in range(len(self.thread)):
            self.test_samples = ['']
            for k in self.prior_test_samples:
                for j in self.thread[i]:      
                    self.test_samples.append(str(k)+str(j))
            self.prior_test_samples = copy.deepcopy(self.test_samples)
        self.steps = []
        for i in self.test_samples:
            if len(i) == len(self.thread):
                self.steps.append(i)               
    def init_path_strands(self):
        for j in self.steps:
            for i in self.types:
                self.dummy_strand[int(i[-1])] = list(j)[self.position_dict[i[-1]]]
            self.path_sequences.append(list_to_string(self.dummy_strand))

class seq_library:
    def __init__(self, data):
        self.data = data
        self.seq_len = self.data[0].seq_len
        self.size = len(data)
        self.mutations = []
        self.matrix_rep = []
        for i in self.data:
            self.matrix_rep.append(i.matrix_rep)
        self.matrix_rep = np.asarray(self.matrix_rep)
    def calc_metrics(self):
        self.percent_A = np.zeros(self.seq_len)
        self.percent_G = np.zeros(self.seq_len)
        self.percent_T = np.zeros(self.seq_len)
        self.percent_C = np.zeros(self.seq_len)
        for i in self.data:
            for j in range(self.seq_len):
                if i.sequence[j] == 'A':
                    self.percent_A[j] += 1
                if i.sequence[j] == 'G':
                    self.percent_G[j] += 1
                if i.sequence[j] == 'C':
                    self.percent_C[j] += 1
                if i.sequence[j] == 'T':
                    self.percent_T[j] += 1
        self.percent_A /= self.size
        self.percent_G /= self.size
        self.percent_C /= self.size
        self.percent_T /= self.size
        
    def add_mutation(self,mutation):
        self.mutations.append(mutation)
    
    def internal_analysis(self,ref_index):
        self.comparisons = [compare_sequences(self.data[ref_index], i) for i in self.data]
               
    def query(self, sequence, verbose = 'N'):
        for i in self.data:
            if sequence == i.sequence:
                if verbose != 'N':
                    print("Sequence Found " + str(sequence))
                return (sequence, i.index, i)
    
    def build_graph(self, start_index, depth=3, limit=0.0, stats_data = 'n/a', save='N', black_list = [], activity_threshold = (0.0, 300.0)):
        self.data[start_index].detect_neighbors(self)
        self.G = nx.Graph()
        self.G.add_node(start_index, color=self.data[start_index].atts['kA'])
        self.next_layer = [start_index]
        self.node_list = [start_index]
        self.master_node_list = []
        self.master_node_list.append(start_index)
        self.network_mutations = []    
        self.net_mut_nosym = []
        self.black_list = black_list
        self.activity_threshold = activity_threshold
        for i in range(depth):
            current_layer_size = len(self.next_layer)
            print("Processing Layer "+str(i))
            print("Size of layer "+str(current_layer_size))
            start_time = time.time()
            self.next_layer = []
            node_count = 0
            for current_node in self.node_list:

                self.data[current_node].detect_neighbors(self)

                for j in self.data[current_node].neighbors:
                    self.G.add_node(j.index, color=j.atts['kA'])
                    self.G.add_edge(current_node, j.index)
                    if (current_node not in self.black_list) and (self.data[current_node].atts['kA'] > self.activity_threshold[0]) and (self.data[current_node].atts['kA'] < self.activity_threshold[1]):
                        self.network_mutations.append((mutation(self.data[current_node], self.data[j.index], stats_lib=stats_data), (self.data[current_node].atts['kA'] - self.data[j.index].atts['kA'])))
                        self.network_mutations.append((self.network_mutations[-1][0].reverse(), self.network_mutations[-1][0].reverse().deltas['kA']))
                        self.net_mut_nosym.append((mutation(self.data[current_node], self.data[j.index], stats_lib=stats_data), (self.data[current_node].atts['kA'] - self.data[j.index].atts['kA'])))
                for m in self.data[current_node].neighbors:
                    if (m.index not in self.master_node_list) == True:
                        self.next_layer.append(m.index)
                        self.master_node_list.append(m.index)
                node_count += 1
            self.node_list = copy.deepcopy(self.next_layer)
            print("Computation time "+str(round((time.time() - start_time), 3)))
        self.G_draw = copy.deepcopy(self.G)
        self.community = copy.deepcopy(list(self.G.nodes))
        
        self.node_colors = []
        for i in list(self.G_draw.nodes):
            if self.G_draw.nodes[i]['color'] >= limit:
                pass
            else:
                self.G_draw.remove_node(i)
        for i in list(self.G_draw.nodes):
            if len(self.G_draw.edges([i])) == 0:
                self.G_draw.remove_node(i)
            else:
                pass
        for i in list(self.G_draw.nodes):
            self.node_colors.append(self.G.nodes[i]['color'])


        if save != 'N':
            nx.draw(self.G_draw, node_color=self.node_colors, with_labels='True', node_size=200.0, font_size=8, cmap='RdYlGn')
            plt.savefig(save)
        self.trim_community = []
        for i in range(len(self.community)):
            if self.data[self.community[i]].atts['kA'] > self.activity_threshold[0] and self.data[self.community[i]].atts['kA'] < self.activity_threshold[1]:
                self.trim_community.append(self.community[i])
        
        self.delta_Kas = [i[1] for i in self.network_mutations]
            
        return self.trim_community
#%%    
class sequence_optimizer:
    def __init__(self, path, data_file, seed, active_threshold = (0.0,300.0), mutation_threshold = 15.0, max_depth = 2, k_split = 3, percent_remove = 0.0, graph_limit = 15.0):
#### Hyperparameters
#### The 'Active Threshold' is the hyperparameter specifying the minimum activity in the database. It is interpretted as the minimum threshold
#### an organism could have and be viable
        self.active_threshold = active_threshold
#### Statistical threshold is the pvalue threshold below which a mutation is considered to be statistically significant. It is sent to
#### 0.05 per the tradition
        self.statistical_threshold = 0.05
#### The reference sequence is the seed of the cluster, from which the sequence space connectivity is mapped from the database
        self.ref_seq = seed
#### The mutation threshold determines how much the activity must change for a mutation to be consider evolutionarily relevant to fitness.
        self.mutation_threshold = mutation_threshold
#### k_split is the K in the K-fold validation process, while also determining the number of SVMs that contribute to the consensus evolution process
        self.k_split = k_split
### I/O parameters
        self.path = path
        self.file = data_file
#### Depth is the distance from the reference sequence used during clustering. For the Chen byo-doped-results data, its maximum safe limit is 2
        self.max_depth = max_depth
### Graph Parameters
        self.graph_limit = graph_limit
######
        self.hyperparameter_dict = {'active_threshold' : self.active_threshold, 'ref_seq' : self.ref_seq, 'mutation_threshold' : self.mutation_threshold, 'k_split' :self.k_split, 'max_depth' : self.max_depth}
###### Sampling Parameters
        self.percent_remove = percent_remove
###
        self.total_data = []

        self.reference_community = pickle.load(open(self.path+"comm_actThr_10_mutThr_15_dep2.pkl","rb"))
        self.reference_community.sort(key=lambda x:x)
        self.reference_community.pop(2)

        if self.percent_remove != 0.0:            
            self.black_list = self.reference_community[-int(self.percent_remove*len(self.reference_community)):]
        else:
            self.black_list = []
       

    def initialize_data(self):
        self.raw_data = pickle.load(open(self.path+self.file, "rb"))
        self.raw_data.sort(key = lambda x: x['pointEstimation']['kA'], reverse = True)
        print("Processing Data") 
##### Reference sequences     
### Reference Central 181 (S-2.1-a) ATTACCCTGGTCATCGAGTGA
### Reference (S-1A.1-a) CTACTTCAAACAATCGGTCTG
        print("Using reference sequence "+self.ref_seq)
        self.sequences = []
        self.entropies = []
        self.lengths = []
        self.kA = []
        self.index = 0
        for i in self.raw_data:   
            if len(i['seq']) == 21 and str(i['pointEstimation']['kA']) != 'nan':
                self.sequences.append(seq(i, index = self.index))
                self.index += 1    
### Initializes Metrics for Each Sequence
        for i in self.sequences:
            i.calc_metrics()
        print("Initializing Library")
        self.lib = seq_library(self.sequences)
        print("Searching for Reference in DB...")
        self.ref_index = self.lib.query(self.ref_seq, verbose = 'Y')[1]
        print("Index of Reference: "+str(self.ref_index))
        ### Calculates Global Library Metrics
        self.lib.calc_metrics()
        self.progress=0.0
        for i in self.lib.data:
            i.learn_library(self.lib)
            self.progress += 1
            # if progress % 10 == 0:
            #     print(str(round(100*progress/len(lib.data),5))+"%")
        print("Library Initiatilized")
            
        del self.raw_data
                
    def build_cluster(self, stats_data = 'n/a'):
        self.reference = self.lib.query(self.ref_seq)[-1]
        self.load_cluster = 'N'
        if self.load_cluster == 'N':
            self.community = self.lib.build_graph(self.reference.index, depth=self.max_depth, limit=self.graph_limit, stats_data=stats_data, save = 'N', activity_threshold = self.active_threshold)
        if self.load_cluster == 'Y':
            self.community = pickle.load(open(self.path+"clustered_nodes_seed_"+str(self.reference.index)+".pkl", 'rb'))
            self.lib.network_mutations = pickle.load(open(self.path+"clustered_edges_seed_"+str(self.reference.index)+".pkl", 'rb'))

        self.seqs = [self.lib.data[i] for i in self.community]
        self.data = []
        self.X = []
        self.Y = []
        
### Mutation Threshold is a hyperparameter ###
        
        for i in self.lib.network_mutations:
            self.X.append((i[0].vectors[0]))
            self.Y.append(grain(i[1], - self.mutation_threshold[0],self.mutation_threshold[0], self.mutation_threshold[1])[0])
        self.community_Kas = [self.lib.data[i].atts['kA'] for i in self.community]

        #Load Master Index List
    def train_svm(self, kernal = 'rbf'):

        self.evolved_sequences = []
        self.mccs = []
        self.met_scores = []
        self.prod = DS.data_set(self.X,self.Y)
        
        self.db_size = (len(self.X) - (len(self.X)%self.k_split))
        print("Using Database of Size "+str(self.db_size))
        self.prod.shuffle()
        self.prod.clip(self.db_size)
        self.prod.k_fold(int(self.db_size/self.k_split))
        for sect in range(self.k_split):
            print("Training SVM "+str(sect+1)+" of "+str(self.k_split))
            self.models = []
            self.X_train, self.Y_train, self.X_test, self.Y_test = self.prod.k_it(sect)
            self.number_attempts = 9
            self.Cs = []
            self.gammas = []
            self.accs = []
            self.root = int(self.number_attempts**(1/2))
            self.acc = 0.0
            self.count = 0
            self.C_candidates = np.arange(0.1,10.1,(10/self.root))
            self.gamma_candidates = np.arange(0.001, 1000.001, (1000/self.root))
            for index_gamma in range(self.root):
                for index_C in range(self.root):
                    self.C = self.C_candidates[index_C]
                    self.gamma = self.gamma_candidates[index_gamma]
                    self.model = DS.SVM(self.X_train, self.Y_train, self.X_test, self.Y_test, self.count, self.C, self.gamma, kernal = kernal)
                    self.model.metrics()       
                    self.count += 1
                    self.Cs.append(self.C)
                    self.gammas.append(self.gamma)
                    self.accs.append(self.model.acc)
                    self.acc = self.model.acc
                    self.models.append(self.model)
                    self.mccs.append(self.model.acc)                     
                        
            # Predict Rules  
            self.models.sort(key = lambda x:x.acc)
            self.prod_model = self.models[-1]
            #
            self.alphabet = ['A', 'C', 'G', 'T']
            self.pred = []
            self.codes = {}
            for k in range(21):
                self.codes.update({k : {}})
                for i in self.alphabet:
                    self.codes[k].update({i : {}})
                    for j in self.alphabet:
                        if i != j:
                            self.pred.append(mutation_code(i,j,k,21))                
                            self.codes[k][i].update({j : self.prod_model.predict([self.pred[-1]])})
                            
            # Express rules as likelihoods
            self.favorables = {}
            for k in range(21):
                self.favorables.update({k : {}})
                for i in self.alphabet:
                    self.favorables[k].update({})
                    for j in self.alphabet:
                        if i != j:               
                            self.favorables[k].update({i : sum(self.codes[k][i].values())[0]})
                            
            # Predict Optimized Sequences                                      
            self.evolved_sequence = {}

            for k in range(21):
                self.values = []
                for i in self.alphabet:
                    if self.favorables[k][i] == max(list(self.favorables[k].values())):
                        self.values.append(i)
                self.evolved_sequence.update({k : self.values})
            
            
            self.sequences_exp = ['']

            self.pred_data = []
            for i in self.sequences_exp:
                self.pred_data.append(seq({'seq' : i}))
                
            self.differences = [compare_sequences(i, self.reference) for i in self.pred_data]
            
            self.met_scores.append( self.models[-1].acc)
            self.evolved_sequences.append(self.evolved_sequence)
            
        self.avg_met = np.average(np.asarray(self.met_scores))
        self.del_met = max(self.met_scores) - min(self.met_scores)
        
        print("The Average Acc is : "+str(round(self.avg_met, 4)))
        print("Delta of Acc is : "+str(round(self.del_met, 4)))
          
    def consensus(self):
        self.library_dictionary = {'A':0, 'C':0, 'G':0, 'T':0}
        self.final = []
        self.final_sequence = []
        for i in range(len(self.evolved_sequences[0])):
            self.final.append(np.asarray([0]*4))
            
        for i in range(len(self.evolved_sequences[0])):
            for j in range(len(self.evolved_sequences)):
                self.final[i] += let2vec(self.evolved_sequences[j][i])
        
        self.pred_seq = {}
        
    def score_seq(self, seq):
        self.score = 0
        for i in range(len(seq)):
            self.score += self.favorables[i][seq[i]]
        return self.score
#%%
def score_seq(favorables, seq):
    score = 0
    for i in range(len(seq)):
        score += favorables[i][seq[i]]
    return score   


    #%% Execution
path = "D:/projects/project-0_skunk/data/RealRNAData/"
data_file = "byo-doped-results.pkl"
stats_file = "welch_t_test.json"
seed = "CCACACTTCAAGCAATCGGTC"
#%%
###   
dict1 = {} 
with open(path+'/'+stats_file, 'r') as myfile:
    raw_data=myfile.read()
    stats = json.loads(raw_data)
for i in stats:
    if i[0] not in dict1.keys():
        dict1.update({i[0] : {}})
for i in stats:
    if i[0] in dict1.keys():
        if i[1] not in dict1[i[0]].keys():
            dict1[i[0]].update({i[1] : i[2]})

#%%
###
db_size = []
number_results = []
top_result = []
trials = {}
activity = {}

def line_pred(Xs, m, b):
    Ys = []
    for i in Xs:
        Ys.append(m*i + b)
    return Ys


### Above 25 mutations threshold = (100.0, 7)
### Above 10 mutation threshold = (100.0, 11)
### Above 0 mutation threshold= (100.0,21)
    
#%%

optimizerHigh = sequence_optimizer(path, data_file, seed,  active_threshold = (0.0,200.00), mutation_threshold = (10.0,11), percent_remove = 0.0, max_depth = 1, graph_limit = 15.0, k_split = 3)
optimizerHigh.initialize_data()
optimizerHigh.build_cluster(dict1)
#%%  
for master_index in range(0, 1):    
    mutations = []
    for n in range(21):
        mutations.append([i[0].deltas['kA'] for i in optimizerHigh.lib.net_mut_nosym if i[0].site == n])
        
    for N in range(21):
        activity.update({N : sum(mutations[N])})
        
    pltX = [i for i in range(21)]
    pltY = [activity[i] for i in range(21)]
     
    below = [optimizerHigh.lib.data[i].atts['kA'] for i in optimizerHigh.community if optimizerHigh.lib.data[i].atts['kA'] < 4.0]
    above = [optimizerHigh.lib.data[i].atts['kA'] for i in optimizerHigh.community if optimizerHigh.lib.data[i].atts['kA'] >= 4.0]
    
    optimizerHigh.k_split = 3
    optimizerHigh.train_svm()
    optimizerHigh.consensus()
    
    #%%
    scores = {}
    full_scores = {}
    class_data = {}
    
    for i in optimizerHigh.community:
        
        optimizerHigh.lib.data[i].atts.update({'score':optimizerHigh.score_seq(optimizerHigh.lib.data[i].sequence)})
        scores.update({i : optimizerHigh.score_seq(optimizerHigh.lib.data[i].sequence )})
    
    for i in optimizerHigh.lib.community:
        if i not in optimizerHigh.community:           
            optimizerHigh.lib.data[i].atts.update({'score':optimizerHigh.score_seq(optimizerHigh.lib.data[i].sequence)})
            full_scores.update({i : optimizerHigh.score_seq(optimizerHigh.lib.data[i].sequence)})
            
    #%%
    
    pickle.dump(optimizerHigh.favorables, open("D:/projects/project-7_RNA/outputs/peak1A/models_above_0/matrix_"+str(master_index)+".pkl", "wb"))   
    #%%
    pltX, pltY = [], []
    f_pltX, f_pltY = [], []
    for i in scores.keys():
        pltX.append(scores[i])
        pltY.append(optimizerHigh.lib.data[i].atts['kA'])    
    for i in full_scores.keys():
        f_pltX.append(full_scores[i])
        f_pltY.append(optimizerHigh.lib.data[i].atts['kA'])
    plt.scatter(pltX, pltY)
    plt.scatter(f_pltX, f_pltY)

    plt.xlabel("Model Score")
    plt.ylabel("Ka")
    plt.savefig("D:/projects/project-7_RNA/outputs/peak1A/figures_above_0/model_"+str(master_index)+".png")
    plt.show()
    
    #%%
    
    
