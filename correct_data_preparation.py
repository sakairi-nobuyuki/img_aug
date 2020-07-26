import os
import glob
import shutil
import re
import xmltodict
import json
import pprint
import pathlib
import xml.etree.ElementTree as ET
import csv

import pandas as pd

def file_read(path):
    with open(path, 'r') as file:
        text = file.read()
    return text

def extract_bboxes_inf (ord_dict):
    sub_dict = {}
    sub_dict['attr'] = ord_dict['name']
    sub_dict['x1'] = ord_dict['bndbox']['xmin']
    sub_dict['y1'] = ord_dict['bndbox']['ymin']
    sub_dict['x2'] = ord_dict['bndbox']['xmax']
    sub_dict['y2'] = ord_dict['bndbox']['ymax']

    return sub_dict

def load_correct_data (correct_data_dir_path, path_str = None):
    #import xml.etree.ElementTree as ET
    correct_list = []
    if path_str == 'jj':
        #xml_path_list = glob.glob ('../../{}/*/*/*/*.xml'.format (correct_data_dir_path), recursive = True)
        #xml_path_list = glob.glob ('{}/train/*/*/*.xml'.format (correct_data_dir_path), recursive = True)
        xml_path_list = glob.glob ('{}/aug/*.xml'.format (correct_data_dir_path), recursive = True)
        #print (xml_path_list)
        for xml_path in xml_path_list:
            dict = xmltodict.parse (file_read (xml_path))
            correct_sub_dict = {}
            #correct_sub_dict['path'] = dict['annotation']['path']
            #correct_sub_dict['path'] = xml_path.replace ('xml', 'jpg')
            correct_sub_dict['path'] = xml_path.replace ('xml', 'png')
    
            annotation_list = []
            if 'object' in dict['annotation'].keys ():
                if type (dict['annotation']['object']) != list:
                    annotation_list.append (extract_bboxes_inf (dict['annotation']['object']))
                else:
                    for sub_dict in dict['annotation']['object']:  
                        annotation_list.append (extract_bboxes_inf (sub_dict))
            correct_sub_dict['annotation'] = annotation_list
            correct_list.append (correct_sub_dict)
        #pprint.pprint (correct_list)
    return (correct_list)


if __name__ == '__main__':
    ### bouding_box = [n_bb, x1, y1, x2, y2, label]
    correct_list = load_correct_data ('annotation', path_str = 'jj')
    #    attr_list = []
    #for correct_data in correct_list:
    #    attr_list.extend[item['attr'] for item in correct_data['annotation']]
    #attr_list = list (set (attr_list))
    #print (attr_list)
    #print (correct_list)

    dataset_data = []
    for correct_dict in correct_list:
        pprint.pprint (correct_dict)
        #img_path = pathlib.Path (correct_dict['path']).relative_to (os.getcwd())
        img_path = correct_dict['path']
        for bboxes in correct_dict['annotation']:
            #if bboxes['attr'] == 'fc'or bboxes:
            if re.match (r'plum', bboxes['attr']):
            #if re.match (r'fc|gl|pu|puf|pcf|auf|ah', bboxes['attr']):                
                print (img_path, int (float (bboxes['x1'])), int (float (bboxes['y1'])), int (float (bboxes['x2'])), int (float (bboxes['y2'])), bboxes['attr'])
                dataset_data.append ([img_path, int (float (bboxes['x1'])), int (float (bboxes['y1'])), int (float (bboxes['x2'])), int (float (bboxes['y2'])), bboxes['attr']])
                #dataset_data.append ([img_path, int (bboxes['x1']), int (bboxes['y1']), int (bboxes['x2']), int (bboxes['y2']), bboxes['attr']])
            #if bboxes['attr'] == 'plumng':
            #    print (img_path, int (float (bboxes['x1'])), int (float (bboxes['y1'])), int (float (bboxes['x2'])), int (float (bboxes['y2'])), bboxes['attr'])
            #    exit ()
                
                #exit ()
    pprint.pprint (dataset_data)

    attr_list = [item[5] for item in dataset_data]
    attr_series = pd.Series (attr_list)
    print (attr_series.value_counts ())

    with open ('dataset.dat', 'w', encoding = 'utf-8', newline = '') as f_out:
        writer = csv.writer (f_out)
        writer.writerows (dataset_data)

        
