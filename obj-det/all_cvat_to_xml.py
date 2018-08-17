import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import os
import dataset_util
import label_map_util
import string
import sys

#initialize height and width of the images extracted from the video
height = 720
width = 1280

def cvat_xml_parser(xml_path, label_map_path):

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

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)#gets the labels from label map
    size = int(root.find('meta').find('task').find('size').text) #parses through xml file to find number of frames
    frames_data = [None] * (size + 1) # the +1 is for interpolation mode

    
    #holds list of frames that will be skipped
    skip = []

    #initializes list of dictionaries

	#each index of list will have a dictionary holding information about the bounding boxes in that frame
    frame_dic = {'xmins' : [], 'xmaxs' : [], 'ymins' : [], 'ymaxs' : [], 
        'classes_text' : [], 'classes' : []} 

    frames_data[i] = [frame_dic for i in range(0,size+1)]

    #parses through the xml file to acquire data from each bounding box
    for track in root.iter('track'):

	#get the label for each track
        label = track.attrib['label'].lower().translate(str.maketrans('','',string.punctuation))
	
	#get all the bboxes in the track and append to the specified index in the list of dictionaries
        for bbox in track.findall('box'):
            if bbox.attrib['outside'] == 0:
                frame_id = int(bbox.attrib['frame'])
                xmin = float(bbox.attrib['xtl'])/width
                xmax = float(bbox.attrib['xbr'])/width
                ymin = float(bbox.attrib['ytl'])/height
                ymax = float(bbox.attrib['ybr'])/height
                class_text = label
                class_ = label_map_dict[class_text]
                class_text = class_text.encode('utf-8')
                frames_data[frame_id]['xmins'].append(xmin)
                frames_data[frame_id]['xmaxs'].append(xmax)
                frames_data[frame_id]['ymins'].append(ymin)
                frames_data[frame_id]['ymaxs'].append(ymax)
                frames_data[frame_id]['classes_text'].append(class_text)
                frames_data[frame_id]['classes'].append(class_)

            if bbox.attrib['occluded'] == 1:
                skip.append(int(bbox.attrib['frame']))

    return frames_data, size, skip

def main(_):
    '''
	Reads the list of dictionaries in the format which is outputted by cvat_xml_parser and 
	writes it as a tf record file. Needs the path to the images extracted from the video of the
	corresponding xml file.
    '''
    #initalize the file paths and the output path

    dataset_dir = 'data/'
    label_map_path = 'classes.pbtxt'
    output_path = 'out.record'

    #write to a tf record file
    writer = tf.python_io.TFRecordWriter(output_path)

    for file in os.listdir('xml_data'):
        xml_path = os.path.join('xml_data/' + file)
        frames_data, size, skip = cvat_xml_parser(xml_path, label_map_path)
        image_path = file[:-4]
        #loops over all the images and writes the image information along with the correspoding bbox info to the example
        for i in range (0, size-1):

            if i in skip:
                continue

            img_path = os.path.join(image_path, '_Image_' + str(i+1).zfill(5) + '.jpg')
            full_path = os.path.join(dataset_dir, img_path).encode('utf-8')

            with tf.gfile.GFile(full_path, 'rb') as fid: #get the encoded img data for each frame
                encoded_jpg = fid.read()

            image_format = 'jpeg'.encode('utf-8')

            data = frames_data[i]

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(full_path),
                'image/source_id': dataset_util.bytes_feature(full_path),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(data['xmins']),
                'image/object/bbox/xmax': dataset_util.float_list_feature(data['xmaxs']),
                'image/object/bbox/ymin': dataset_util.float_list_feature(data['ymins']),
                'image/object/bbox/ymax': dataset_util.float_list_feature(data['ymaxs']),
                'image/object/class/text': dataset_util.bytes_list_feature(data['classes_text']),
                'image/object/class/label': dataset_util.int64_list_feature(data['classes']),
            }))


            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    tf.app.run()
