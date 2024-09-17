"""
This file reads the synthetic dataset and creates new labels as a function of old labels

Usage:
python create_new_labels.py --data <dataset_folder> --num_new_labels <num_new_labels> --rule_input_size <rule_input_size> --data_noise <data_noise>
"""

import numpy as np 
import arff
import re
from itertools import product
import random
random.seed(0)
import argparse
import os
import pickle
from collections import defaultdict
import copy
from sklearn.model_selection import train_test_split

# Useful for saving the rules of the labels
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def write_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def binary_representation_to_num(binary_list):
    """
    """
    power_array = 2** np.asarray(range(len(binary_list)))
    return np.dot(binary_list, power_array)

def create_new_labels(x, y, num_labels, rule_input_size = 2, num_new_labels=3):
    """
    """
    all_possible_rules = list(product([0,1], repeat = 2**rule_input_size))
    new_labels_data = []
    rules = [] 
    rule_cache = {}
    for j, new_label in enumerate(range(num_new_labels)):
        valid_label = False
        while(not valid_label):
            input_ys = random.sample(range(num_labels), rule_input_size)
            valid_rule = False
            while(not valid_rule):
                rule = random.sample(all_possible_rules,1)[0]
                if not (np.mean(rule) == 1 or np.mean(rule) == 0):
                    valid_rule = True

            new_label_y = []
            for i in range(len(x)):
                y_input_values = y[i][input_ys]
                created_y = rule[binary_representation_to_num(y_input_values)]
                new_label_y.append(created_y)
            
            new_label_mean = np.mean(new_label_y)
            if new_label_mean > 0.05 and new_label_mean<=0.8:
                valid_label = True

        print(f"New label {j} stats, mean: {new_label_mean}")
        new_labels_data.append(new_label_y)
        rules.append({'rule': rule, 'input_y': input_ys})

    y_new_labels = np.vstack(new_labels_data).T 
    y_final = np.hstack((y,y_new_labels))
    return y_final, rules 


def load_data(filename):
    """
    """
    dataset = arff.load(open(filename, 'r'))
    num_labels = 0 
    for i, attr in enumerate(dataset['attributes']):
        if attr[0][0] == 'y':
            num_labels+=1
    data = np.array(dataset['data'])
    x = np.asarray(data[:,:-num_labels])
    y = np.asarray(data[:,-num_labels:])
    x = x.astype('float')
    y = y.astype('int')
    return x, y, num_labels, dataset

def prepare_new_dataset(args):
    """
    """
    if args.data_noise == 0:
        filename = os.path.join(args.data, 'DataBase.arff')
    else:
        filename = os.path.join(args.data, f'DataBase_noise_{args.data_noise}.arff')
    
    print(f"Reading data from {filename}")

    x, y, num_labels, dataset = load_data(filename)
    # dataset.keys() : ['description', 'relation', 'attributes', 'data']

    # print(dataset.keys())
    # exit()
    y_final, rules = create_new_labels(x=x,y=y, num_labels=num_labels, rule_input_size= args.rule_input_size, num_new_labels = args.num_new_labels)
    # print(y_final.shape)


    for elem in rules:
        print(f"Rule : {elem['rule']}, input y: {elem['input_y']}")

    # Save the rules as pickle
    rules_pickle_file = filename[:-5] + f'_rule_input_size_{args.rule_input_size}_num_new_labels_{args.num_new_labels}_rules.pickle'
    write_pickle(rules, rules_pickle_file)

    # Split data into train, val and test
    x_train, x_test, y_train, y_test = train_test_split(x, y_final, test_size = int(0.2*len(x)), random_state=0) 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = int(0.1*len(x)), random_state=0) 
    
    splitted_data = {'train': [x_train, y_train], 'val': [x_val, y_val], 'test': [x_test, y_test]}

    # Prepare and save the new arff 
    # train / val / test
    for data_split_type in ['train', 'val', 'test']:
        x_prime, y_prime = splitted_data[data_split_type]
        x_prime = x_prime.tolist()
        y_prime = y_prime.tolist()

        attributes_list = [x_prime[i] + y_prime[i] for i in range(len(x_prime))]
        # attributes_list = [[str(elem) for elem in point] for point in attributes_list]

        train_dataset = copy.deepcopy(dataset)
        train_dataset['data'] = attributes_list
        train_dataset['description'] += f'\n Num new labels added by rules: {args.num_new_labels} \n Rule input size: {args.rule_input_size} \n Rules saved at: {rules_pickle_file} '
        for i in range(args.num_new_labels):
            train_dataset['attributes'].append((f'y{num_labels+i+1}', ['0', '1']))
        
        output_arff_file_train = filename[:-5] + f'_rule_input_size_{args.rule_input_size}_num_new_labels_{args.num_new_labels}_{data_split_type}.arff'
        with open(output_arff_file_train,'w') as f:
            arff.dump(train_dataset,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--num_new_labels', type=int, default=5)
    parser.add_argument('--rule_input_size', type=int, default=2)
    # parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--data_noise', type=float, default=0)

    args = parser.parse_args()
    # Check for validity of arguments
    if args.data == '':
        print("Please specify correct data folder")
        exit()
    if args.num_new_labels <=0:
        print("num_new_labels must be greater than 0")
        exit()
    # if args.output_dir == '':
    #     print("Please specify correct output folder")
    #     exit()
    if args.data_noise not in [0, 0.1, 0.05]:
        print("Please specify correct data noise (0 or 0.05 or 0.1)")
        exit()


    prepare_new_dataset(args)
