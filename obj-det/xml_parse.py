import xml.etree.ElementTree as ET

tree = ET.parse('19_andy.xml')
root = tree.getroot()


def create_pascal_xml_structure():
    '''Create the structure in which we want the annotation data to be in'''
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder')

    ET.SubElement(annotation, 'filename')

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database')
    ET.SubElement(source, 'annotation')
    ET.SubElement(source, 'image')


    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width')
    ET.SubElement(size, 'height')
    ET.SubElement(size, 'depth')

    ET.SubElement(annotation, 'segmented')

    object = ET.SubElement(annotation, 'object')
    ET.SubElement(object, 'name')
    ET.SubElement(object, 'pose')
    ET.SubElement(object, 'truncated')
    ET.SubElement(object, 'occluded')
    bnd_box = ET.SubElement(object, 'bnd_box')
    ET.SubElement(bnd_box, 'xmin')
    ET.SubElement(bnd_box, 'ymin')
    ET.SubElement(bnd_box, 'xmax')
    ET.SubElement(bnd_box, 'ymax')
    ET.SubElement(object, 'difficult')



    return annotation

#new_root = create_pascal_xml_structure()
#ET.dump(new_root)

def get_annotation_box():
    '''Returns a list of dictionaries of frame number, truncated, xmin, xmax, ymin, ymax of all the boxes annotated in xml file'''
    boxes = []
    label = [x.attrib['label'] for x in root.findall('track')]
    for box in root.iter('box'):
        box.attrib.pop('keyframe')
        box.attrib['truncated'] = box.attrib.pop('outside')
        box.attrib['xmin'] = box.attrib.pop('xtl')
        box.attrib['ymin'] = box.attrib.pop('ytl')
        box.attrib['xmax'] = box.attrib.pop('xbr')
        box.attrib['ymax'] = box.attrib.pop('ybr')

        boxes.append(box.attrib)

    return label



print(int(root.find('meta').find('task').find('size').text))

