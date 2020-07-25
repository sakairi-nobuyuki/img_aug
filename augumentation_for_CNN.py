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

def obtain_mod_path (target_dir_path, decoration, original_path):
    return os.path.join (target_dir_path, decoration + os.path.basename (original_path))


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
        A = aug_conf.A_sharpness[trans_flag % 2]
        print ("  flag is {}. A = {}.".format (trans_flag, A))

        ker = np.array (A, dtype=np.float32)
        self.img = cv2.filter2D (self.img, -1, ker)
        self.modify_img_path ('A_sharp_{}_'.format (A[1][1]))
        self.modify_xml_path ('A_sharp_{}_'.format (A[1][1]))
        
    def add_blur (self, trans_flag):
        print ("in img blur, trans_flag", trans_flag)
        if 'sharp' in self.img_path:
            print ('  sharpness already done. nothing can be done with img blur. going to abort the process.')
            return 0
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
        alpha = aug_conf.alpha_list[trans_flag_alpha]
        gamma = aug_conf.gamma_list[trans_flag_gamma]

        print ("  alpha = {}, gamma = {}".format (alpha, gamma))

        self.img = self.img * alpha + gamma

        self.modify_img_path ('alpha_{}_gamma_{}_'.format (alpha, gamma))
        self.modify_xml_path ('alpha_{}_gamma_{}_'.format (alpha, gamma))

    def add_gaussian_noize (self, trans_flag_sigma):
        print ("in adding Gaussian noize")    
        sigma = aug_conf.sigma_list[trans_flag_sigma]
        print ("  sigma = {}".format (sigma))
        row,col,ch= self.img.shape
        mean = 0
        gauss = np.random.normal (mean, sigma, (row,col,ch))
        gauss = gauss.reshape (row, col, ch)
        self.img = self.img + gauss
        
        self.modify_img_path ('gaussian_sigma_{}_'.format (sigma))
        self.modify_xml_path ('gaussian_sigma_{}_'.format (sigma))  

    def offset_img (self, trans_flag_dx, trans_flag_dy):
        print ("in offsetting image")
        dx = trans_flag_dx
        dy = trans_flag_dy
        print ("  dx = {}, dy = {}.".format (dx, dy))
        
        img_w, img_h, _ = self.img.shape

        center = (int (0.5 * img_w), int (0.5 * img_h))
        trans = np.array ([[1, 0, dx], [0, 1, dy]], dtype = np.float32)
        print ("  transform matrix:")
        pprint.pprint (trans, indent = 2)

        self.img = cv2.warpAffine(self.img, trans, (img_w, img_h))

        self.modify_img_path ('dx_{}_dy_{}_'.format (dx, dy))
        self.modify_xml_path ('dx_{}_dy_{}_'.format (dx, dy))

        for bb, bb_mod in zip (self.xml_obj.bboxes, self.xml_obj.bboxes_mod):
            bb_mod['x1'] = int (bb['x1']) + dx
            bb_mod['y1'] = int (bb['y1']) + dy
            bb_mod['x2'] = int (bb['x2']) + dx
            bb_mod['y2'] = int (bb['y2']) + dy
        #self.xml_obj.save_geometrical_modification ('', '', self.xml_path)

    def rotate_img (self, target_flag_dtheta):
        print ("in rotating image")
        dtheta = aug_conf.rotation_deg[target_flag_dtheta]
        if 'dx' in self.img_path or 'dy' in self.img_path:
            print ("  offset already done. rotation cannot be carried out.")
            return 0
        print ("  dheta = {}".format (dtheta))
        img_w, img_h, _ = self.img.shape

        center = (int (0.5 * img_w), int (0.5 * img_h))

        scale = (img_h + img_w * np.sin (np.radians (abs (dtheta)))) / img_h
        trans = cv2.getRotationMatrix2D(center, dtheta , scale)
        self.img = cv2.warpAffine(self.img, trans, (img_w, img_h))

        self.modify_img_path ('dtheta_{}_'.format (dtheta))
        self.modify_xml_path ('dtheta_{}_'.format (dtheta))

        bb_xml = BboxesInXML (xml_path)    
        for bb, bb_mod in zip (self.xml_obj.bboxes, self.xml_obj.bboxes_mod):
        #for bb, bb_mod in zip (bb_xml.bboxes, bb_xml.bboxes_mod):
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

    def save_geometrical_modification (self, target_dir_path, decorator, new_path = None):
        #new_path = os.path.join (target_dir_path, self.original_dir, decorator + self.original_name)
        if new_path == None:
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

        n_aug = np.random.randint (0, aug_conf.max_n_aug)
        print ("For {}, {} times augumentation shall be carried out.".format (os.path.basename (img_path), n_aug))

        ### 乱数でaugmentationするしないと、パラメータを選ぶ。
        for i_aug in range (0, n_aug):
            ### augumentation用のオブジェクトを定義する
            aug_obj = AugumentationObjects (aug_img_path, resized_img_path, resized_xml_path)
            print ("init augumentation object")

            print ("the {} th augumentation for {}".format (i_aug, os.path.basename (img_path)))
            ###  シャープネス
            if  np.random.randint (0, aug_conf.prob_like_sharpness) != 0:
                aug_obj.sharpen_img (np.random.randint (0, len (aug_conf.A_sharpness)))
        
            ###  ボケ        
            if  np.random.randint (0, aug_conf.prob_like_blur) != 0:
                aug_obj.add_blur (np.random.randint (0, len (aug_conf.k_blur)))

            ###  彩度を変える
            if  np.random.randint (0, aug_conf.prob_like_colour_tone) != 0:
                aug_obj.modify_colour_tone (np.random.randint (0, len (aug_conf.s_mag_list)), np.random.randint (0, len (aug_conf.v_mag_list)))

            ###  コントラストを変える
            if  np.random.randint (0, aug_conf.prob_like_contrast) != 0:
                aug_obj.modify_contrast (np.random.randint (0, len (aug_conf.alpha_list)), np.random.randint (0, len (aug_conf.gamma_list)))

            ###  ノイズを加える
            if  np.random.randint (0, aug_conf.prob_like_noize) != 0:
                aug_obj.add_gaussian_noize (np.random.randint (0, len (aug_conf.sigma_list)))

            ###  オフセットする  (±10, 20, 40)
            if  np.random.randint (0, aug_conf.prob_like_offset) != 0:
                aug_obj.offset_img (np.random.randint (0, len (aug_conf.offset_dx)), np.random.randint (0, len (aug_conf.offset_dy)))

            ### 回転させる
            if  np.random.randint (0, aug_conf.prob_like_rot) != 0:
                aug_obj.rotate_img (np.random.randint (0, len (aug_conf.rotation_deg)))

            ### 画像とxmlを保存する
            ### xmlは、xmlのデータがaffine transformで変わるのと、その保存は別にやらないといけない。！！！！
            cv2.imwrite (aug_obj.img_path, aug_obj.img)
            aug_obj.xml_obj.save_geometrical_modification ('', '', aug_obj.xml_path)
        exit ()



        




