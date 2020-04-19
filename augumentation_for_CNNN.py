# coding: utf-8
import os
import glob
import pprint
import shutil
import math

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict

import aug_conf

H_SIZE = 256

def obtain_imgs_path_list (img_dir_path = None):
    #img_path_list = ['annotation/test/nanase-01.jpg']
    img_path_list = glob.glob ('annotation/original/**/*.jpg')
    pprint.pprint (img_path_list)
    return img_path_list

def obtain_imgs_and_xml_path_list (img_dir_path = None):
    #img_path_list = ['annotation/test/nanase-01.jpg']
    if img_dir_path != None:
        img_path_list = glob.glob ('{}/**/*.jpg'.format (img_dir_path), recursive = True)
        xml_path_list = [item.replace ('jpg', 'xml') for item in img_path_list]
    else:
        img_path_list = glob.glob ('./annotation/original/**/*.jpg', recursive = True)
        xml_path_list = [item.replace ('jpg', 'xml') for item in img_path_list]

    #pprint.pprint (img_path_list)
    #print ("open: ", img_path_list[0], os.path.exists (img_path_list[0]))
    return img_path_list, xml_path_list


def modify_colour_tone (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    #img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
    print ("modifying color tone of {}".format (img_path))
    s_mag_list = [1.0, 0.7]   ## magnitude of saturation
    v_mag_list = [1.0, 0.7]   ## magnitude of value
    #v_mag_list = [1.0]   ## magnitude of value
    #print ("{} of which shape {} shall be modified".format (img_path, img_hsv.shape))

    img_mod_list = []
    img_path_mod_list = []
    xml_path_mod_list = []
    for s_mag in s_mag_list:
        for v_mag in v_mag_list:
            img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
            if float (s_mag) == 1.0 and  float (v_mag) == 1.0:  continue
            img_hsv_mod = img_hsv
            img_hsv_mod[:, :, (1)] = img_hsv[:, :, (1)] * s_mag
            
            img_hsv_mod[:, :, (2)] = img_hsv[:, :, (2)] * v_mag
            print ("s_mag: {}, v_mag: {}, s: {}, v: {}".format (s_mag, v_mag, img_hsv_mod[:, :, (1)].mean, img_hsv_mod[:, :, (2)].mean))
            img_mod = cv2.cvtColor (img_hsv_mod, cv2.COLOR_HSV2BGR)
            img_path_mod = os.path.join (target_dir_path, 's_mag_{}_v_mag_{}_'.format (s_mag, v_mag) + os.path.basename (img_path))
            xml_path_mod = os.path.join (target_dir_path, 's_mag_{}_v_mag_{}_'.format (s_mag, v_mag) + os.path.basename (xml_path))
            #img_path_mod = os.path.join (os.path.dirname (img_path).replace ('test', 'aug'), 's_mag_{}_v_mag_{}_'.format (s_mag, v_mag) + os.path.basename (img_path))
            #xml_path_mod = os.path.join (os.path.dirname (img_path).replace ('test', 'aug'), 's_mag_{}_v_mag_{}_'.format (s_mag, v_mag) + os.path.basename (img_path).replace ('jpg', 'xml'))
            img_mod_list.append (img_mod)
            #cv2.imshow ("test", img_mod)
            #cv2.waitKey (0)
            #cv2.destroyAllWindows ()
            #img_path_mod_list.append (img_path_mod)
            #xml_path_mod_list.append (xml_path_mod)
            cv2.imwrite (img_path_mod, img_mod)
            shutil.copy (xml_path, xml_path_mod)
    
    

def rezise_img_and_xml (target_dir_path, img_path, xml_path):
    if os.path.exists (target_dir_path) == False:
        print ('cannot find {}'.format (target_dir_path))
        os.makedirs (target_dir_path)

    print (img_path)
    if len (img_path.split ('\\')) - len (target_dir_path.split ('\\')) > 0:
        #interrim_name = ['{}_'.format (item) for item in img_path.split ('\\') [len (target_dir_path.split ('\\')):]][0]
        interrim_name = ''.join (['{}_'.format (item) for item in img_path.split ('\\') [1:-1]])
    else:
        None
    print ("interrim name: ", interrim_name)

    ### resize img and save
    print ("img path: ", img_path)
    img = cv2.imread (img_path)
    w, h, c = img.shape
    scale = H_SIZE / h
    print ("img size: ({}, {}), scale: {}".format (w, h, scale))
    img_path_mod = os.path.join (target_dir_path, interrim_name + 'resize_' + os.path.basename (img_path))
    img_mod = cv2.resize (img, fx = scale,  fy = scale, dsize = None)
    cv2.imwrite (img_path_mod, img_mod)

    ### modify xml
    print ("xml path: ", xml_path)
    tree = ET.parse (xml_path)
    root = tree.getroot ()
    annotation_objects = root.findall ('object')
    for annotation_object in annotation_objects:
        print (annotation_object.find ('name').text)
        bboxes =  (annotation_object.find ('bndbox'))
        obj_sizes = ['xmin', 'xmax', 'ymin', 'ymax']
        ### modify geometrical parameters
        for obj_size in obj_sizes:
            print (obj_size, bboxes.find (obj_size).text)
            bboxes.find (obj_size).text = str (int (scale * int (bboxes.find (obj_size).text)))
            print (obj_size, bboxes.find (obj_size).text)
    ### save xml
    xml_path_mod = os.path.join (target_dir_path, interrim_name + 'resize_' + os.path.basename (xml_path))
    tree.write (xml_path_mod)    

    return img_path_mod, xml_path_mod

def add_gaussian_noize (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    row,col,ch= img.shape
    mean = 0

    sigma_list = [1, 2, 5, 10]
    
    for sigma in sigma_list:
        gauss = np.random.normal (mean, sigma, (row,col,ch))
        gauss = gauss.reshape (row, col, ch)
        img_mod = img + gauss
        img_path_mod = obtain_mod_path (target_dir_path, 'sigma_{}'.format (sigma), img_path)
        xml_path_mod = obtain_mod_path (target_dir_path, 'sigma_{}'.format (sigma), xml_path)

        cv2.imwrite (img_path_mod, img_mod)
        shutil.copy (xml_path, xml_path_mod)

def obtain_mod_path (target_dir_path, decoration, original_path):
    return os.path.join (target_dir_path, decoration + os.path.basename (original_path))

def modify_contrast (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)

    alpha_list = [0.6, 1.0, 1.2, 1.4]
    gamma_list = [-20, 0, 20, 50]

    img_mod_list = []
    img_path_mod_list = []
    for alpha in alpha_list:
        for gamma in gamma_list:
            img_mod = img * alpha + gamma
            #img_path_mod = os.path.join (os.path.dirname (img_path).replace ('test', 'aug'), 'alpha_{}_gamma_{}_'.format (alpha, gamma) + os.path.basename (img_path))
            img_path_mod = os.path.join (target_dir_path, 'alpha_{}_gamma_{}_'.format (alpha, gamma) + os.path.basename (img_path))
            xml_path_mod = os.path.join (target_dir_path, 'alpha_{}_gamma_{}_'.format (alpha, gamma) + os.path.basename (xml_path))
            cv2.imwrite (img_path_mod, img_mod)            
            shutil.copy (xml_path, xml_path_mod)

def obtain_rotation_ndarray (theta):
    theta = np.radians (theta)
    return np.array ([[np.cos (theta), -1.0 * np.sin (theta)], \
        [np.sin (theta), np.cos (theta)]])

def obtain_rotated_pos (x1, y1, x0, y0, theta):
    r1 = np.array ([float (x1), float (y1)]).reshape ((2, 1))
    r0 = np.array ([float (x0), float (y0)]).reshape ((2, 1))
    #print ("r0 = \n  {}\n\n r1 = \n  {}".format (r0, r1))
    #print ("r1 - r0", r1 - r0)
    
    A  = obtain_rotation_ndarray (theta)
    #print (A)
    r2 = np.dot (A, r1 - r0) + r0
    #print (np.dot (A, r1 - r0))
    #print ("rotation from {} to {} by {}deg".format (r1, r2, theta))
    return r2

def rotate_img (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    img_w, img_h, _ = img.shape
    bb_xml = BboxesInXML (xml_path)

    center = (int (0.5 * img_w), int (0.5 * img_h))

    dtheta_list = aug_conf.rotation_deg

    print (dtheta_list)
    
    for dtheta in dtheta_list:
        scale = (img_h + img_w * np.sin (np.radians (abs (dtheta)))) / img_h
        trans = cv2.getRotationMatrix2D(center, dtheta , scale)
        img_mod = cv2.warpAffine(img, trans, (img_w, img_h))
        img_path_mod = obtain_mod_path (target_dir_path, 'dtheta_{}_'.format (dtheta), img_path)
        print ("img size original: ", img_w, img_h)
        print ("transformation:    ", trans)
        print ("img size modified: ", img_mod.shape[1], img_mod.shape[0])
        cv2.imwrite (img_path_mod, img_mod)

        ### xmlの編集
        for bb, bb_mod in zip (bb_xml.bboxes, bb_xml.bboxes_mod):
            r1 = np.array ([bb['x1'], bb['y1']], dtype=float)
            r2 = np.array ([bb['x2'], bb['y2']], dtype=float)
            #print ("r1:     ", r1)
            #print ("Atrans: ")
            #pprint.pprint (trans)
            #print ("Atrans minor: ")
            trans_A_minor = trans[0:2, 0:2]
            trans_A_add   = trans[0:2, 2:3]
            pprint.pprint (trans_A_minor)
            print ("Atrans add: ")
            pprint.pprint (trans_A_add)
            
            r1_mod = np.dot (trans_A_minor, r1.T) + trans_A_add.T
            r2_mod = np.dot (trans_A_minor, r2.T) + trans_A_add.T

            print ("r1: {}\nr1_mod: {}".format (r1, r1_mod))
            print ("r2: {}\nr2_mod: {}".format (r2, r2_mod))

            print (bb['attr'])
            x1s = [bb['x1'], bb['x2']]
            y1s = [bb['y1'], bb['y2']]
            x_cand = []
            y_cand = []
            print ("bb    ", bb, (int (bb['x2'])  - int (bb['x1'])) * (int (bb['y2'])  - int (bb['y1'])))
            for x1 in x1s:
                for y1 in y1s:
                    #print ('x0: {}, y0: {}'.format (center[0], center[1]))
                    #print ('x1: {}, y1: {}'.format (x1, y1))
                    x2, y2 = obtain_rotated_pos (x1, y1, center[0], center[1], dtheta)
                    x_cand.append (int (x2[0]))
                    y_cand.append (int (y2[0]))
                    
            bb_mod['x1'] = min (x_cand)
            bb_mod['y1'] = min (y_cand)
            bb_mod['x2'] = max (x_cand)
            bb_mod['y2'] = max (y_cand)
            
            print ("bb_mod", bb_mod, (int (bb_mod['x2'])  - int (bb_mod['x1'])) * (int (bb_mod['y2'])  - int (bb_mod['y1'])))
        print ("modified img path: ", img_path_mod)
        bb_xml.save_geometrical_modification (target_dir_path, 'dtheta_{}_'.format (dtheta))
            

    exit ()



def offset (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    img_w, img_h, _ = img.shape
    bb_xml = BboxesInXML (xml_path)

    dx_list = aug_conf.offset_dx
    dy_list = aug_conf.offset_dy
    
    for dx in dx_list:
        for dy in dy_list:
            for bb_mod, bb in zip (bb_xml.bboxes_mod, bb_xml.bboxes):
                if bb['x1'] + dx < 0:         break
                if bb['x2'] + dx > bb_xml.w:  break
                if bb['y1'] + dy < 0:         break
                if bb['y2'] + dy > bb_xml.h:  break
                bb_mod['x1'] = bb['x1'] + dx
                bb_mod['x2'] = bb['x2'] + dx
                bb_mod['y1'] = bb['y1'] + dy
                bb_mod['y2'] = bb['y2'] + dy
            bb_xml.w_mod = bb_xml.w - dx
            bb_xml.h_mod = bb_xml.h - dy
    



def save_mod_img_and_xml (img_mod_list, img_path_mod_list, img_path_list, xml_path_mod_list):
    for img_mod, img_path_mod in zip (img_mod_list, img_path_mod_list):
        cv2.imwrite (img_path_mod, img_mod)



class BboxesInXML:
    def __init__(self, xml_path):
        tree = ET.parse (xml_path)
        root = tree.getroot ()
        annotation_objects = root.findall ('object')
        self.NN = len (annotation_objects)
        self.original_dir  = root.find ('folder').text
        self.original_path = xml_path
        self.original_name = os.path.basename (xml_path)
        self.original_name_in_annotation = root.find ('filename').text
        self.modified_name = ''

        self.w = root.find ('size').find ('width').text
        self.h = root.find ('size').find ('height').text
        
        self.w_mod = self.w
        self.h_mod = self.h

        self.bboxes = []
        self.bboxes_mod = []
        for annotation_object in annotation_objects:
            bboxes =  (annotation_object.find ('bndbox'))
            bbox_dict = {}
            bbox_dict['attr'] = annotation_object.find ('name').text
            bbox_dict['x1'] = bboxes.find ('xmin').text
            bbox_dict['y1'] = bboxes.find ('ymin').text
            bbox_dict['x2'] = bboxes.find ('xmax').text
            bbox_dict['y2'] = bboxes.find ('ymax').text
            #pprint.pprint (bbox_dict)
            self.bboxes.append (bbox_dict)
            self.bboxes_mod.append (bbox_dict)
    def scale (self, img_h):
        rate = img_h / self.h
        self.bboxes_mod = []

        for bbox in self.bboxes:
            bbox_dict = {}
            bbox_dict['attr'] = bbox['attr']
            bbox_dict['x1'] = bbox['x1'] * rate
            bbox_dict['y1'] = bbox['y1'] * rate
            bbox_dict['x2'] = bbox['x2'] * rate
            bbox_dict['y2'] = bbox['y2'] * rate
        self.bboxes_mod.append (bbox_dict)

    def save_geometrical_modification (self, target_dir_path, decorator):
        #new_path = os.path.join (target_dir_path, self.original_dir, decorator + self.original_name)
        new_path = os.path.join (target_dir_path, decorator + self.original_name)
        print ("oririnal path:            ", self.original_path)
        print ("new path to save changes: ", new_path)
        tree = ET.parse (self.original_path)
        root = tree.getroot ()
        annotation_objects = root.findall ('object')
        for annotation_object, bbox_mod in zip (annotation_objects, self.bboxes_mod):
            bboxes =  (annotation_object.find ('bndbox'))
            bboxes.find ('xmin').text = str (bbox_mod['x1'])
            bboxes.find ('xmax').text = str (bbox_mod['x2'])
            bboxes.find ('ymin').text = str (bbox_mod['y1'])
            bboxes.find ('ymax').text = str (bbox_mod['y2'])
        root.find ('size').find ('width').text  = str (self.w_mod)
        root.find ('size').find ('height').text = str (self.h_mod)
        tree.write (new_path)

if __name__ == '__main__':
    '''
    データがデカかったら縮小して、適当にargumentationする。
    各関数は[{img_path: 'img_path', xml_path: 'xml_path'}, ...]を返し、関数内でimg_pathに画像を保存して、xmlも編集する。
    '''
    ### オリジナルの学習ファイルを読み込む
    #img_path_list = obtain_imgs_path_list ()
    img_path_list, xml_path_list = obtain_imgs_and_xml_path_list ('./annotation/original')
    
    ### 実際に学習に使うディレクトリを決める。
    aug_img_path = './annotation/aug'
    if os.path.exists (aug_img_path) == False:  os.makedirs (aug_img_path)

    for img_path, xml_path in zip (img_path_list, xml_path_list):
        ### サイズがデカかったら小さくする。
        resized_img_path, resized_xml_path = rezise_img_and_xml ('./annotation/resize', img_path, xml_path)
        shutil.copy (resized_img_path, aug_img_path)
        shutil.copy (resized_xml_path, aug_img_path)
    
        ### 乱数でaugmentationの数を選ぶようにする

        ###  角度を変える (±5°)
        rotate_img (aug_img_path, resized_img_path, resized_xml_path)


        ###  彩度を変える
        modify_colour_tone (aug_img_path, resized_img_path, resized_xml_path)
                
        ###  コントラストを変える
        modify_contrast (aug_img_path, resized_img_path, resized_xml_path)
        
        ###  ノイズを加える
        add_gaussian_noize (aug_img_path, resized_img_path, resized_xml_path)
        
        ###  オフセットする  (±10, 20, 40)

        ### ぼかす


        

        ###  画像を歪ませる
    
    #### 画像リストを作る

        




