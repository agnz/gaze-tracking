import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import os
import dataset_util
import label_map_util
import string

def cvat_xml_parser(xml_path, label_map_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    size = int(root.find('meta').find('task').find('size').text)
    frames_data = [None] * size

    for i in range (0, size):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        frame_dic = {'xmins' : xmins, 'xmaxs' : xmaxs, 'ymins' : ymins, 'ymaxs' : ymaxs, 
            'classes_text' : classes_text, 'classes' : classes}
        frames_data[i] = frame_dic

    for track in root.iter('track'):
        label = track.attrib['label'].lower().translate(None, string.punctuation)
	
        for bbox in track.findall('box'):
            frame_id = int(bbox.attrib['frame'])
            xmin = float(bbox.attrib['xtl'])
            xmax = float(bbox.attrib['xbr'])
            ymin = float(bbox.attrib['ytl'])
            ymax = float(bbox.attrib['ybr'])
            class_text = label
            class_ = label_map_dict[class_text]
            frames_data[frame_id]['xmins'].append(xmin)
            frames_data[frame_id]['xmaxs'].append(xmax)
            frames_data[frame_id]['ymins'].append(ymin)
            frames_data[frame_id]['ymaxs'].append(ymax)
            frames_data[frame_id]['classes_text'].append(class_text)
            frames_data[frame_id]['classes'].append(class_)

    return frames_data, size

def main(_):

    xml_path = '19_andy.xml'
    dataset_dir = 'data'
    data_dir = 'Andy'
    output_path = 'out.record'
    label_map_path = 'pascal_label_map'
    height = 1280
    width = 720

    frames_data, size = cvat_xml_parser(xml_path, label_map_path)

    writer = tf.python_io.TFRecordWriter(output_path)

    for i in range (1, size, 2):
        img_path = os.path.join(data_dir, '_Image_' + str(i).zfill(5) + '.jpg')
        full_path = os.path.join(dataset_dir, img_path)
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        image_format = 'jpeg'

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
