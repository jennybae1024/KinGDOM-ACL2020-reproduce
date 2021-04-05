from tqdm import tqdm
import numpy as np
import os.path, pickle
from utils import obtain_all_seed_concepts
from utils_graph import conceptnet_graph, domain_aggregated_graph, subgraph_for_concept
import argparse

pkl_path = '/media/disk1/jennybae/data/kingdom/pkl_files'

filename={"conceptnet": "conceptnet_english.txt",
          "wordnet18": "wordnet18.txt"}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_name', type=str, default='conceptnet')
    parser.add_argument('--dataset_type', type=str, default='data2000')
    args = parser.parse_args()
    pkl_path = os.path.join(pkl_path, args.dataset_type, args.kg_name)
    bow_size = 5000

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    print ('Extracting seed concepts from all domains.')
    all_seeds = obtain_all_seed_concepts(bow_size, args.dataset_type)
    
    print ('Creating conceptnet graph.')
    G, G_reverse, concept_map, relation_map = conceptnet_graph(filename[args.kg_name])
    
    print ('Num seed concepts:', len(all_seeds))
    print ('Populating domain aggregated sub-graph with seed concept sub-graphs.')
    triplets, unique_nodes_mapping = domain_aggregated_graph(all_seeds, G, G_reverse, concept_map, relation_map)
    
    print ('Creating sub-graph for seed concepts.')
    concept_graphs = {}

    for node in tqdm(all_seeds, desc='Instance', position=0):
        concept_graphs[node] = subgraph_for_concept(node, G, G_reverse, concept_map, relation_map)
        
    # Create mappings
    inv_concept_map = {v: k for k, v in concept_map.items()}
    inv_unique_nodes_mapping = {v: k for k, v in unique_nodes_mapping.items()}
    inv_word_index = {}
    for item in inv_unique_nodes_mapping:
        inv_word_index[item] = inv_concept_map[inv_unique_nodes_mapping[item]]
    word_index = {v: k for k, v in inv_word_index.items()}
        
    print ('Saving files.')
        
    pickle.dump(all_seeds, open(os.path.join(pkl_path, 'all_seeds.pkl'), 'wb'))
    pickle.dump(concept_map, open(os.path.join(pkl_path,'concept_map.pkl'), 'wb'))
    pickle.dump(relation_map, open(os.path.join(pkl_path, 'relation_map.pkl'), 'wb'))
    pickle.dump(unique_nodes_mapping, open(os.path.join(pkl_path, 'unique_nodes_mapping.pkl'), 'wb'))
    pickle.dump(word_index, open(os.path.join(pkl_path, 'word_index.pkl'), 'wb'))
    pickle.dump(concept_graphs, open(os.path.join(pkl_path, 'concept_graphs.pkl'), 'wb'))
    
    np.ndarray.dump(triplets, open(os.path.join(pkl_path, 'triplets.np'), 'wb'))
    print ('Completed.')