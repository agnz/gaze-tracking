import numpy as np
import xml.etree.ElementTree as ET
import os
import string
import sys

height = 720
width = 1280

def cvat_xml_parser(xml_path): #label_map_path):

    '''
	:param: xml_path, label_map_path
	:output: frames_data, size
	
	Finds the path to the xml file outputted by CVAT annotation tool and parses through.
	Organizes the bounding box coordinates by xmin, ymin, xmax, class_text, and class number, which
	is then put in a dictionary sorting the arrays into a dictionary for each frame. The dictionary is
	then put into an array consisting of all the frames ordered by frame number.
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()

    #label_map_dict = label_map_util.get_label_map_dict(label_map_path)#gets the labels from label map
    size = int(root.find('meta').find('task').find('size').text) #parses through xml file to find number of frames
    frames_data = [None] * (size + 1) # the +1 is for interpolation mode

    #initializes list of dictionaries
    for i in range (0, size + 1):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

	#each index of list will have a dictionary holding information about the bounding boxes in that frame
        frame_dic = {'xmins' : xmins, 'xmaxs' : xmaxs, 'ymins' : ymins, 'ymaxs' : ymaxs, 
            'classes_text' : classes_text, 'classes' : classes} 

        frames_data[i] = frame_dic

    #parses through the xml file to acquire data from each bounding box
    for track in root.iter('track'):

	#get the label for each track
        label = track.attrib['label'].lower().translate(str.maketrans('','',string.punctuation))
	
	#get all the bboxes in the track and append to the specified index in the list of dictionaries
        for bbox in track.findall('box'):
            frame_id = int(bbox.attrib['frame'])
            xmin = float(bbox.attrib['xtl'])/width
            xmax = float(bbox.attrib['xbr'])/width
            ymin = float(bbox.attrib['ytl'])/height
            ymax = float(bbox.attrib['ybr'])/height
            class_text = label
            #class_ = label_map_dict[class_text]
            class_text = class_text.encode('utf-8')
            frames_data[frame_id]['xmins'].append(xmin)
            frames_data[frame_id]['xmaxs'].append(xmax)
            frames_data[frame_id]['ymins'].append(ymin)
            frames_data[frame_id]['ymaxs'].append(ymax)
            frames_data[frame_id]['classes_text'].append(class_text)
            #frames_data[frame_id]['classes'].append(class_)

    return frames_data, size

