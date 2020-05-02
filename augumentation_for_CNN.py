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

def sharpen_img (aug_obj, trans_flag):
#def sharpen_img (target_dir_path, img_path, xml_path, trans_flag):
    print ("trans_flag", trans_flag)
    if trans_flag == 0:
        return aug_obj
    else:
        A = aug_conf.A_sharpness[trans_flag % 2]

    ker = np.array (A, dtype=np.float32)
    aug_obj['img'] = cv2.filter2D (aug_obj['img'], -1, ker)
    img_path_mod = os.path.join (target_dir_path, 'A_sharp_{}_'.format (A[1][1]) + os.path.basename (img_path))
    xml_path_mod = os.path.join (target_dir_path, 'A_sharp_{}_'.format (A[1][1]) + os.path.basename (xml_path))
    cv2.imwrite (img_path_mod, img_mod)            
    shutil.copy (xml_path, xml_path_mod)        

    exit ()

    img = cv2.imread (img_path)

    for A in aug_conf.A_sharpness:
        ker = np.array (A, dtype=np.float32)
        img_mod = cv2.filter2D (img, -1, ker)
        img_path_mod = os.path.join (target_dir_path, 'A_sharp_{}_'.format (A[1][1]) + os.path.basename (img_path))
        xml_path_mod = os.path.join (target_dir_path, 'A_sharp_{}_'.format (A[1][1]) + os.path.basename (xml_path))
        cv2.imwrite (img_path_mod, img_mod)            
        shutil.copy (xml_path, xml_path_mod)
    exit ()

def add_blur (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    img_w, img_h, _ = img.shape

    for k_blur in aug_conf.k_blur:
        img_mod = cv2.blur (img, ksize = (k_blur, k_blur))
        img_path_mod = os.path.join (target_dir_path, 'k_blur_{}_'.format (k_blur) + os.path.basename (img_path))
        xml_path_mod = os.path.join (target_dir_path, 'k_blur_{}_'.format (k_blur) + os.path.basename (xml_path))
        cv2.imwrite (img_path_mod, img_mod)            
        shutil.copy (xml_path, xml_path_mod)
    


def rotate_img (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    img_w, img_h, _ = img.shape
    

    center = (int (0.5 * img_w), int (0.5 * img_h))

    dtheta_list = aug_conf.rotation_deg

    print (dtheta_list)
    
    for dtheta in dtheta_list:
        scale = (img_h + img_w * np.sin (np.radians (abs (dtheta)))) / img_h
        trans = cv2.getRotationMatrix2D(center, dtheta , scale)
        img_mod = cv2.warpAffine(img, trans, (img_w, img_h))
        img_path_mod = obtain_mod_path (target_dir_path, 'dtheta_{}_'.format (dtheta), img_path)
        #print ("img size original: ", img_w, img_h)
        #print ("transformation:    ", trans)
        #print ("img size modified: ", img_mod.shape[1], img_mod.shape[0])
        cv2.imwrite (img_path_mod, img_mod)
        img_test     = img
        img_mod_test = img_mod
        ### xmlの編集
        bb_xml = BboxesInXML (xml_path)    
        for bb, bb_mod in zip (bb_xml.bboxes, bb_xml.bboxes_mod):
            #### r1, r2, r3, r4は左上から時計回りに長方形の頂点座標を示す。
            r1 = np.array ([bb['x1'], bb['y1']], dtype=float)
            r2 = np.array ([bb['x2'], bb['y1']], dtype=float)
            r3 = np.array ([bb['x2'], bb['y2']], dtype=float)
            r4 = np.array ([bb['x1'], bb['y2']], dtype=float)

            ### 回転返還の行列はopenCVでゲットしたやつを流用する。面倒なので。
            trans_A_minor = trans[0:2, 0:2]
            trans_A_add   = trans[0:2, 2:3]
                        
            r1_mod = np.dot (trans_A_minor, r1.T) + trans_A_add.T * scale
            r2_mod = np.dot (trans_A_minor, r2.T) + trans_A_add.T * scale
            r3_mod = np.dot (trans_A_minor, r3.T) + trans_A_add.T * scale
            r4_mod = np.dot (trans_A_minor, r4.T) + trans_A_add.T * scale
                        
            bb_mod['x1'] = np.max (np.mean ([r1_mod[0][0], r4_mod[0][0]]), 0)
            bb_mod['y1'] = np.max (np.mean ([r1_mod[0][1], r2_mod[0][1]]), 0)
            bb_mod['x2'] = np.max (np.mean ([r2_mod[0][0], r3_mod[0][0]]), 0)
            bb_mod['y2'] = np.max (np.mean ([r3_mod[0][1], r4_mod[0][1]]),  0)

        bb_xml.save_geometrical_modification (target_dir_path, 'dtheta_{}_'.format (dtheta))



def offset_img (target_dir_path, img_path, xml_path):
    img = cv2.imread (img_path)
    img_w, img_h, _ = img.shape

    center = (int (0.5 * img_w), int (0.5 * img_h))
    
    for dx in aug_conf.offset_dx:
        for dy in aug_conf.offset_dy:
            #scale = (img_h + img_w * np.sin (np.radians (abs (dtheta)))) / img_h
            trans = np.array ([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
            
            print (trans)
            
            img_mod = cv2.warpAffine(img, trans, (img_w, img_h))
            img_path_mod = obtain_mod_path (target_dir_path, 'dx_{}_dy_{}_'.format (dx, dy), img_path)
            #print ("transformation:    ", trans)
            #print ("img size modified: ", img_mod.shape[1], img_mod.shape[0])
            cv2.imwrite (img_path_mod, img_mod)
            img_test     = img
            img_mod_test = img_mod
            ### xmlの編集
            bb_xml = BboxesInXML (xml_path)    
            for bb, bb_mod in zip (bb_xml.bboxes, bb_xml.bboxes_mod):
                bb_mod['x1'] = int (bb['x1']) + dx
                bb_mod['y1'] = int (bb['y1']) + dy
                bb_mod['x2'] = int (bb['x2']) + dx
                bb_mod['y2'] = int (bb['y2']) + dy

            bb_xml.save_geometrical_modification (target_dir_path, 'dx_{}_dy_{}_'.format (dx, dy))
    

    



def save_mod_img_and_xml (img_mod_list, img_path_mod_list, img_path_list, xml_path_mod_list):
    for img_mod, img_path_mod in zip (img_mod_list, img_path_mod_list):
        cv2.imwrite (img_path_mod, img_mod)

def init_aug_obj (img_path, xml_path):
    aug_obj = {}
    aug_obj['img'] = cv2.imread (img_path)
    aug_obj['xml'] = BboxesInXML (xml_path)
    aug_obj['img_path'] = img_path
    aug_obj['xml_path'] = xml_path
    return aug_obj

def get_trans_flag ():
    return np.random.randint (0, 5)

class AugumentationObjects:
    def __init__(self, target_dir_path, img_path, xml_path):
        self.img = cv2.imread (img_path)
        self.img_path = img_path

        self.xml_obj  = BboxesInXML (xml_path)
        self.xml_path = xml_path

        self.target_dir_path = target_dir_path

    def modify_img_path (self, attr):
        self.img_path = os.path.join (self.target_dir_path, '{}_'.format (attr) + os.path.basename (self.img_path))
    
    def modify_xml_path (self, attr):
        self.xml_path = os.path.join (self.target_dir_path, '{}_'.format (attr) + os.path.basename (self.xml_path))

    def sharpen_img (self, trans_flag):
        print ("in sharpening img, trans_flag", trans_flag)
        if trans_flag == 0:
            print ("  flag is 0, and nothing to do.")
            return 0
        else:
            A = aug_conf.A_sharpness[trans_flag % 2]
            print ("  flag is {}. A = {}.".format (trans_flag, A))

        ker = np.array (A, dtype=np.float32)
        self.img = cv2.filter2D (self.img, -1, ker)
        self.modify_img_path ('A_sharp_{}_'.format (A[1][1]))
        self.modify_xml_path ('A_sharp_{}_'.format (A[1][1]))

        return 1
        
    def add_blur (self, trans_flag):
        print ("in img blur, trans_flag", trans_flag)
        if 'sharp' in self.img_path:
            print ('  sharpness already done. nothing can be done with img blur. going to abort the process.')
            return 0
        if trans_flag == 0:
            print ("  flag is 0, and nothing to do.")
            return 0
        else:
            k_blur = aug_conf.k_blur[trans_flag % len (aug_conf.k_blur)]
            print ("  flag is {}. k_blur = {}.".format (trans_flag, k_blur))

        self.img = cv2.blur (self.img, ksize = (k_blur, k_blur))
        self.modify_img_path ('k_blur_{}_'.format (k_blur))
        self.modify_xml_path ('k_blur_{}_'.format (k_blur))


    def modify_colour_tone (self, trans_flag_s_mag, trans_flag_v_mag):
        print ("in modifying color tone")
        s_mag = aug_conf.s_mag_list[trans_flag_s_mag]
        v_mag = aug_conf.v_mag_list[trans_flag_v_mag]
        print ("  s magnitude = {}, v magnitude = {}".format (s_mag, v_mag))

        img = self.img
        img_hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
        img_hsv_mod = img_hsv
        img_hsv_mod[:, :, (1)] = img_hsv[:, :, (1)] * s_mag
            
        img_hsv_mod[:, :, (2)] = img_hsv[:, :, (2)] * v_mag
        
        self.img = cv2.cvtColor (img_hsv_mod, cv2.COLOR_HSV2BGR)

        self.modify_img_path ('s_mag_{}_v_mag_{}_'.format (s_mag, v_mag))
        self.modify_xml_path ('s_mag_{}_v_mag_{}_'.format (s_mag, v_mag))

    def modify_contrast (self, trans_flag_alpha, trans_flag_gamma):
        print ("in modifying contrast")
        if trans_flag_alpha == 0 or trans_flag_gamma == 0:
            print ("  trans flag == 0, nothing to do")
            return 0
        else:
            alpha = aug_conf.alpha_list[trans_flag_alpha]
            gamma = aug_conf.gamma_list[trans_flag_gamma]
            print ("  alpha = {}, gamma = {}".format (alpha, gamma))

        self.img = self.img * alpha + gamma

        self.modify_img_path ('alpha_{}_gamma_{}_'.format (alpha, gamma))
        self.modify_xml_path ('alpha_{}_gamma_{}_'.format (alpha, gamma))

    def add_gaussian_noize (self, trans_flag_sigma):
        print ("in adding Gaussian noize")    
        if trans_flag_sigma == 0:
            print ("  Gaussian sigma == 0, nothing can be done")
            return 0
        else:
            sigma = aug_conf.sigma_list[trans_flag_sigma]
            print ("  sigma = {}".format (sigma))
        row,col,ch= self.img.shape
        mean = 0
        gauss = np.random.normal (mean, sigma, (row,col,ch))
        gauss = gauss.reshape (row, col, ch)
        self.img = self.img + gauss
        
        self.modify_img_path ('gaussian_sigma_{}_'.format (sigma))
        self.modify_xml_path ('gaussian_sigma_{}_'.format (sigma))  



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
    画像の返還はランダムっぽくやる。

    色々な変換に乱数trans_flagを渡す。
    transformation (aug_img_path, img_path, xml_path, trans_flag)

    変換の中で、本当に変換するしない、するならそのときのパラメータを決める。
    def transformation (aug_img_path, img_path, xml_path, trans_flag):
        if trans_flag == 0:
            return img, xml, img_path, xml_path
        else:
            trans_parameter = function_to_make_parameter (trans_flag)
        img processing and xml processing...

        return img, xml, img_path_to_save, xml_path_to_save
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

        ### augumentation用のオブジェクトを定義する
        aug_obj = AugumentationObjects (aug_img_path, resized_img_path, resized_xml_path)

        ### 乱数でaugmentationの数を選ぶようにする

        
        ###  シャープネス
        aug_obj.sharpen_img (0)
        
        ###  ボケ        
        aug_obj.add_blur (2)
        ###  彩度を変える
        aug_obj.modify_colour_tone (1, 2)

        ###  コントラストを変える
        aug_obj.modify_contrast (1, 2)

        ###  ノイズを加える
        aug_obj.add_gaussian_noize (2)

        cv2.imwrite (aug_obj.img_path, aug_obj.img)
        exit ()

        

        ###  オフセットする  (±10, 20, 40)
        offset_img (aug_img_path, resized_img_path, resized_xml_path)

        ###  角度を変える (±5°)
        rotate_img (aug_img_path, resized_img_path, resized_xml_path)

        
        

        

        ###  画像を歪ませる
    
    #### 画像リストを作る

        



