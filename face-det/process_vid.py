import sys
import numpy as np
from skimage import transform as tf
import warnings
import os.path
import imageio
import dlib
from PIL import Image, ImageDraw, ImageFont, ImageColor
import datetime
import time
import scipy
from scipy.spatial.distance import euclidean as euclid_dist
from itertools import combinations
import h5py
import names
from clustering import _chinese_whispers

HDF5_DETECTIONS_GROUP = '/detections'
HDF5_TUBELETS_GROUP = '/tubelets'


def import_vid(vid_fname: str, dataset_fname: str = None, msg_freq: int = 100):
    """
    Take a video-file name and run face detector on frame-by-frame basis. The detected face bounding boxes and facial
    landmarks are written into a hdf5 dataset file.

    :param vid_fname: Input video filename
    :param dataset_fname: Output dataset filename (default: vid_fname.h5)
    :param msg_freq: Report progress every msg_freq frames. Reporting disabled if == 0
    """

    # if dataset filename is not specified, use video location and name
    input_path = os.path.abspath(os.path.expanduser(vid_fname))
    if not dataset_fname:
        dataset_fname = os.path.splitext(vid_fname)[0] + '.h5'

    # load the dlib detector models
    cnn_face_detector_path = 'models/mmod_human_face_detector.dat'
    predictor_path = 'models/shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'
    assert os.path.exists(cnn_face_detector_path), "Please download {} model.".format(cnn_face_detector_path)
    assert os.path.exists(predictor_path), "Please download {} model.".format(predictor_path)
    assert os.path.exists(face_rec_model_path), "Please download {} model.".format(face_rec_model_path)
    cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)
    shape_predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    reader = imageio.get_reader(input_path)

    # We increase the accuracy of the face bounding box by detecting 5 facial landmarks, as described below.
    # The bounding box is then defined by these landmarks locations.

    # face gets warped with a similarity transform to the flowing canonical form:
    #  p0 == left eye corner, outside part of eye.
    #  p1 == left eye corner, inside part of eye.
    #  p2 == right eye corner, outside part of eye.
    #  p3 == right eye corner, inside part of eye.
    #  p4 == immediately under the nose, right at the top of the philtrum.
    landmark_dst = np.array([[0.8595674595992, 0.2134981538014],  # p0
                             [0.6460604764104, 0.2289674387677],  # p1, etc
                             [0.1205750620789, 0.2137274526848],
                             [0.3340850613712, 0.2290642403242],
                             [0.4901123135679, 0.6277975316475],
                             ])
    # face chip extraction parameters as used by dlib's compute_face_descriptor()
    chip_size = 150
    chip_padding = 0.25
    # calculate canonical landmark locations within the chip_size x chip_size image
    for p in range(landmark_dst.shape[0]):
        landmark_dst[p, 0] = (chip_padding + landmark_dst[p, 0]) / (2 * chip_padding + 1) * chip_size
        landmark_dst[p, 1] = (chip_padding + landmark_dst[p, 1]) / (2 * chip_padding + 1) * chip_size

    # set-up the dataset structure
    dataset = {'vid_fname': vid_fname,
               'num_frames': len(reader),
               'fps': reader.get_meta_data()['fps'],
               'import_datetime': str(datetime.datetime.now()).split('.')[0],
               'detector': 'dlib cnn',
               'shape_predictor': 'dlib 5 point',
               'face_rec': 'dlib cnn',
               'descriptor_dim': 128,
               'data': {}  # per-frame data array
               }
    print("Processing {} ({} frames) with {} detector, {} shape predictor and {} face recognition model.".format(
        dataset['vid_fname'],
        dataset['num_frames'],
        dataset['detector'],
        dataset['shape_predictor'],
        dataset['face_rec']
    ))

    tt = time.time()

    # these lines are purely for debugging - ignore
    #    if k > 1000:
    #        break
    # for k in [520, 521, 522, 523]:
    #     img = reader.get_data(k)

    # read the video 1 frame at a time (k = frame index, img = frame contents)
    for frame_id, img in enumerate(reader):
        # Find all the face detections in the current frame of video
        dets = cnn_face_detector(img, 1)

        dataset['data'][frame_id] = []
        for d in dets:
            # d = detection: d.confidence, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()

            # Detect 5 facial landmarks in image chip defined by d.rect bounding box
            shape = shape_predictor(img, d.rect)
            # compute the face descriptor of the face aligned to canonical pose (internally by dlib)
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # compute the warp of the face to this canonical pose (replicates dlib's internals - no python wrapper)
            landmark_src = np.zeros((5, 2))
            for p in range(shape.num_parts):
                landmark_src[p, 0] = shape.part(p).x
                landmark_src[p, 1] = shape.part(p).y

            tform = tf.SimilarityTransform()
            tform.estimate(landmark_src, landmark_dst)
            bbox_src = np.array([[0, 0],
                                 [0, chip_size],
                                 [chip_size, chip_size],
                                 [chip_size, 0],
                                 ])
            # 4 bbox corners' locations in image after alignment to canonical pose
            # storing all 4 corners as bbox may have been rotated
            bbox_aligned = tform.inverse(bbox_src)

            # Similarity transform 3x3 matrix
            bbox_tform = tform.params

            # [top-left.x top-left.y bottom-right.x bottom-right.y]
            bbox = (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())

            dataset['data'][frame_id].append({'bbox': bbox,
                                              'confidence': d.confidence,
                                              'bbox_aligned': bbox_aligned.tolist(),
                                              'bbox_tform': bbox_tform.tolist(),
                                              'landmarks': landmark_src.tolist(),
                                              'descriptor': list(face_descriptor),
                                              'label': -1,
                                              'label_str': ''})

        # report progress every msg_freq frames
        if msg_freq > 0 and frame_id % msg_freq == 0:
            processing_fps = msg_freq / (time.time() - tt)
            tt = time.time()
            print("Processed {} frames, {} frames remaining. Current speed {:.1f} fps. Approx local finish time {}.".format(
                frame_id,
                len(reader)-frame_id,
                processing_fps,
                str(datetime.datetime.now() + datetime.timedelta(seconds=(len(reader) - frame_id)/processing_fps)).split('.')[0]
            ))

    filter_double_detections(dataset)

    save_dataset(dataset, dataset_fname)

    print("Finished. Saved to {}".format(dataset_fname))

    return dataset_fname, dataset

def filter_double_detections(dataset: dict) -> dict:
    """
    Dlib cnn face detector sometimes detects the same face twice:
    * One correct detection with a tight bbox with high confidence ~1
    * Another one with bbox double the size, centred in the same place and very low conf <0.2
    * NMS supression fails as IOU < 0.5 (their thresh) because of such difference in size
    * Examples: vid001_recompressed_60fps.mp4 frames 521 and 523
    Need to filter these, either by proximity in coords to existing bboxes with high conf or by proximity in descriptor
    (or both)
    This function performs this filtering

    :param dataset:
    :return: Filtered dataset
    """
    # consider two faces to be identical if Euclid(desc1,desc2) < THRESH (matching descriptor within tolerance)
    DESC_TOL = 0.4

    # consider two faces identical if distance between their centers < XY_THRESH*bbox_side
    XY_THRESH = 0.5

    data = dataset['data']

    # loop over all video frames
    for frame_id in range(dataset['num_frames']):
        if frame_id not in data:
            continue
        cur_frame = data[frame_id]  # current frame's detections data

        # get descriptors of all detections in the current frame into an array
        cur_descriptors = []
        for det in cur_frame:
            cur_descriptors.append(det['descriptor'])
        cur_descriptors = np.array(cur_descriptors)

        # find all pairs of matching descriptors
        matches = []
        for i1, i2 in combinations(range(len(cur_frame)), 2):
            if euclid_dist(cur_frame[i1]['descriptor'], cur_frame[i2]['descriptor']) < DESC_TOL:
                # if two faces are very similar in their descriptors
                # check distance between centre's of aligned bboxes vs tolerance
                xy_dist = euclid_dist(
                    np.array(cur_frame[i1]['bbox_aligned']).mean(axis=0),
                    np.array(cur_frame[i2]['bbox_aligned']).mean(axis=0))
                # calculate tolerance as a fraction of first bbox's side length
                # (bboxes are square and hopefully relatively similar in size)
                xy_tol = euclid_dist(
                    cur_frame[i1]['bbox_aligned'][0],
                    cur_frame[i1]['bbox_aligned'][1]) * XY_THRESH
                if xy_dist < xy_tol:
                    # print("DEBUG: dist {:.1f}, tol {:.1f}".format(xy_dist, xy_tol))
                    matches.append((i1, i2))

        # remove detection with lower confidence
        for i1, i2 in matches:
            remove_idx = i1 if cur_frame[i1]['confidence'] < cur_frame[i2]['confidence'] else i2
            del cur_frame[remove_idx]

        if len(matches) > 0:
            print("Filtered {} detection(s) from frame {}".format(len(matches), frame_id))

    return dataset


def annotate_vid(dataset: dict, out_vid_fname: str = '', fps: int = 0, limit: int = -1):
    """
    Annotate video with face detections (bounding boxes, labels and confidence)

    :param dataset: Dataset containing the detections
    :param out_vid_fname: Annotated video filename. {input_fname}_annotated.mp4 if not specified
    :param fps: frame rate of annotated video. Input video frame rate will be used if not specified
    :param limit: Only process first limit frames if specified
    """

    vid_fname = dataset['vid_fname']

    if not out_vid_fname:
        fname, _ = os.path.splitext(vid_fname)
        out_vid_fname = fname + '_annotated.mp4'

    input_path = os.path.abspath(os.path.expanduser(vid_fname))
    output_path = os.path.abspath(os.path.expanduser(out_vid_fname))

    reader = imageio.get_reader(input_path)

    if limit == -1:
        limit = len(reader)

    if not fps:
        fps = reader.get_meta_data()['fps']

    fnt = ImageFont.truetype('Times New Roman.ttf', size=25)

    writer = imageio.get_writer(output_path, fps=fps)

    data = dataset['data']
    for frame_id, img in enumerate(reader):
        if frame_id > limit:
            break
        img_ = Image.fromarray(img)
        draw = ImageDraw.Draw(img_, 'RGBA')
        draw.text((10, 10), "Frame #{:d}".format(frame_id), font=fnt, fill='yellow')

        if frame_id not in data:
            print(
                "Warning, frame {} is not present in JSON file. May have not been processed or wrong video...".format(
                    frame_id))
            continue
        dets = data[frame_id]
        # loop over all the face detections in the current video frame
        for det in dets:
            bbox_aligned = det['bbox_aligned']

            # create a semi-transparent color based on detection confidence level
            confidence = det['confidence']
            transparency_scale = 0.75
            color = ImageColor.getrgb('yellow') + (int(255 * confidence * transparency_scale),)

            # draw the aligned bbox into the image
            bbox_aligned = [tuple(l) for l in bbox_aligned]  # convert to list of tuples as required by draw
            draw.polygon(bbox_aligned, outline=color)
            # draw the original bbox into the frame
            # draw.rectangle(det['bbox'], outline='red')

            # if the detection has been assigned a label, print it into the image
            label = det['label_str'].decode("utf-8")
            if not label:
                label = ' '  # setting to ' ' instead of '' due to bug in pillow
            draw.text((bbox_aligned[0][0] + 2, bbox_aligned[0][1]), "{}".format(label), font=fnt,
                      fill=color)
            # if confidence below 0.8, print it into the frame for debugging
            if confidence < 0.8:
                draw.text((bbox_aligned[0][0] + 2, bbox_aligned[0][1]+20),
                          "{:.1f}".format(confidence),
                          font=fnt,
                          fill='yellow')
            # draw the facial landmarks into the image
            for landmark in det['landmarks']:
                draw.ellipse([(landmark[0] - 1, landmark[1] - 1), (landmark[0] + 1, landmark[1] + 1)],
                             outline=color)

        del draw
        writer.append_data(np.array(img_))

    writer.close()


def track_faces(dataset):
    """
    Matches face detections in consecutive video frames to track detections in time and outputs a set of tubelets.
    A tubelet is a list of (frame_id, det_id) tuplets that correspond to the same face over time.
    We want the algorithm to create tubelets in a way that faces within each tubelet belong to the same individual with
    high confidence. Resulting tubelets may be short - but we can cluster them into smaller number of identities later.
    :param dataset:
    :return: dataset with tubelets added
    """
    # TODO figure out a better way to threshold displacement between frames
    # constant dist in pix may not be best approach
    XY_THRESH = 50  # pix
    data = dataset['data']

    tubelets = []  # list of all tubelets
    prev_xy = np.array([])
    prev_frame_tubs = []  # list of tubelets present in the previous frame
    # loop over all video frames
    for frame_id in range(dataset['num_frames']):
        cur_frame_dets = data[frame_id]  # face detections in the current frame

        # calculate (x,y) coordinates of each bbox in the current frame
        # cur_descriptors = []
        cur_xy = []  # for each detected bbox in cur_frame, (x,y) coordinates of bbox centre
        for det in cur_frame_dets:
            # cur_descriptors.append(det['descriptor'])
            bbox = np.array(det['bbox'])
            cur_xy.append((bbox[0:2] + bbox[2:]) / 2)
            # TODO calculate centre of aligned bbox
            # cur_xy.append(np.array(det['bbox_aligned']).mean(axis=0))

        # cur_descriptors = np.array(cur_descriptors)
        cur_xy = np.array(cur_xy)

        if not prev_xy.size or not cur_xy.size:
            # if no detections in prev_frame or cur_frame - no matches can exist
            matches = np.array([-1]*len(cur_frame_dets))
        else:
            # pair-wise distance matrix
            # dist[0,:]: distance from 0th face in cur_frame to every face in prev_frame, etc
            dist = scipy.spatial.distance.cdist(cur_xy, prev_xy, 'Euclidean')

            # naive matching - without removal (may result in many-to-1 matches)
            # for each cur_frame bbox find closest bbox in prev frame
            # TODO incorporate descriptor and constrain to 1-to-1 (unique) matches. Important for large head movement.
            matches = dist.argmin(axis=1)
            # remove any matches that are too far away from each other in xy
            matches[np.where(dist.min(axis=1) > XY_THRESH)] = -1

        # assign matched tubelets from prev_frame to cur_frame
        cur_frame_tubs = [[] for _ in cur_frame_dets] # a list of empty lists same size as cur_frame_dets
        for cur_frame_det_idx, prev_frame_det_idx in enumerate(matches):
            if prev_frame_det_idx != -1:
                # if match was found - assign the corresponding tubelet from previous frame
                cur_frame_tubs[cur_frame_det_idx] = prev_frame_tubs[prev_frame_det_idx]
            else:
                # if no match found
                # start a new tubelet
                tubelets.append(cur_frame_tubs[cur_frame_det_idx])
            cur_frame_tubs[cur_frame_det_idx].append((frame_id, cur_frame_det_idx))

        prev_frame_tubs = cur_frame_tubs
        prev_xy = cur_xy

    dataset['tubelets'] = tubelets

    return dataset


def tubelets_mean_descriptors(dataset):
    """
    Computes mean descriptor for each tubelet by averaging descriptors of all detections within a tubelet.
    :param dataset:
    :return: n_tubs x descriptor_dim numpy array
    """
    data = dataset['data']
    tubelets = dataset['tubelets']
    # calculate average/mean descriptor for each tubelet
    tub_mean_descriptors = []
    for label, tubelet in enumerate(tubelets):
        # collect descriptors for all detections in the tubelet and average them
        tub_descriptors = []
        for frame_id, det_id in tubelet:
            tub_descriptors.append(data[frame_id][det_id]['descriptor'])
        tub_descriptors = np.array(tub_descriptors)
        tub_mean_descriptors.append(tub_descriptors.mean(axis=0))

    tub_mean_descriptors = np.array(tub_mean_descriptors)

    return tub_mean_descriptors


def tubelet_overlap(dataset):
    """
    Compute overlap of tubelets in time. If two tubelets contain detections in the same video frame - they overlap.
    Useful a priori information for clustering. Overlapping tubelets should not be in the same cluster.
    :param dataset:
    :return: n_tubs x n_tubs array where 0 indicates no overlap, 1 overlap
    """
    tubelets = dataset['tubelets']
    # tubelet overlap: a square matrix representing which tubelets overlap with each other in time (frame)
    tubelet_overlap = np.diag([1] * len(tubelets))
    for n, p in combinations(range(len(tubelets)), 2):
        tubelet_n_frames, _ = zip(*tubelets[n])
        tubelet_n_frames = set(tubelet_n_frames)
        tubelet_p_frames, _ = zip(*tubelets[p])
        tubelet_p_frames = set(tubelet_p_frames)
        if tubelet_n_frames.intersection(tubelet_p_frames):
            tubelet_overlap[n, p] = 1
            tubelet_overlap[p, n] = 1

    return tubelet_overlap


def export_tubelet(dataset, tubelet, tubelet_fname='tubelet.mp4', fps=-1, expand=0):
    """
    Exports the given tubelet as a video

    :param dataset: dataset
    :param tubelet: tubelet to export
    :param tubelet_fname: filename for the exported video
    :param fps: framerate of the video. Framerate of original video is used if not specified
    :param expand: expand each bbox by this many pixels on each side (sometimes useful for verification)
    """
    vid_fname = dataset['vid_fname']
    data = dataset['data']

    input_path = os.path.abspath(os.path.expanduser(vid_fname))
    output_path = os.path.abspath(os.path.expanduser(tubelet_fname))

    reader = imageio.get_reader(input_path)
    if fps == -1:
        fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for frame_id, det_id in tubelet:
            img = reader.get_data(frame_id)
            T = np.array(data[frame_id][det_id]['bbox_tform'])
            T[0, 2] += expand
            T[1, 2] += expand
            tform = tf.SimilarityTransform(T)
            out_sz = 150 + expand*2
            out_sz = ((out_sz // 16) + 1) * 16  # round up output size to be multiple of 16 (vid compression blk_size)
            img_warped = tf.warp(img, tform.inverse, output_shape=(out_sz, out_sz))
            # increased output shape slightly to be a multiple of 16 (compression block size)
            writer.append_data(img_warped)
    writer.close()


def export_tubelet_images(dataset, tubelet, output_path='tubelet', expand=0):
    from skimage.io import imsave
    """
    Exports the given tubelet as images

    :param dataset: dataset
    :param tubelet: tubelet to export
    :param output_path: path to place the images
    :param expand: expand each bbox by this many pixels on each side (sometimes useful for verification)
    """
    vid_fname = dataset['vid_fname']
    data = dataset['data']

    input_path = os.path.abspath(os.path.expanduser(vid_fname))
    output_path = os.path.abspath(os.path.expanduser(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    reader = imageio.get_reader(input_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for frame_id, det_id in tubelet:
            img = reader.get_data(frame_id)
            T = np.array(data[frame_id][det_id]['bbox_tform'])
            T[0, 2] += expand
            T[1, 2] += expand
            tform = tf.SimilarityTransform(T)
            out_sz = 150 + expand * 2
            img_warped = tf.warp(img, tform.inverse, output_shape=(out_sz, out_sz))
            imsave(os.path.join(output_path, "{:06d}_{:d}.jpg".format(frame_id, det_id)),
                   img_warped
                   )


def load_dataset(dataset_fname):
    file = h5py.File(dataset_fname, 'r')
    dataset = dict(file.attrs)
    data = dict(file[HDF5_DETECTIONS_GROUP])
    data = {int(frame_id): list(dets) for frame_id, dets in data.items()}
    dataset['data'] = data

    tubelets = []
    try:
        for idx in range(len(file[HDF5_TUBELETS_GROUP])):
            tubelets.append(list(file[HDF5_TUBELETS_GROUP + '/' + str(idx)]))
    except KeyError:
        pass  # no tubelets in the file
    dataset['tubelets'] = tubelets

    return dataset


def save_dataset(dataset: dict, dataset_fname: str):
    """

    :param dataset:
    :param dataset_fname:
    """
    file = h5py.File(dataset_fname, 'w')

    # save metadata
    file.attrs['vid_fname'] = dataset['vid_fname']
    file.attrs['num_frames'] = dataset['num_frames']
    file.attrs['fps'] = dataset['fps']
    file.attrs['import_datetime'] = dataset['import_datetime']
    file.attrs['detector'] = dataset['detector']
    file.attrs['shape_predictor'] = dataset['shape_predictor']
    file.attrs['face_rec'] = dataset['face_rec']
    file.attrs['descriptor_dim'] = dataset['descriptor_dim']
    file.attrs['datatype_version'] = 1  # version of the composite type below. Increment if it changes.

    # compound datatype to represent a numpy array of detections (in each frame)
    comp_type = np.dtype([(('bounding box', 'bbox'), '<f4', 4),
                          ('confidence', '<f4'),
                          (('aligned bounding box', 'bbox_aligned'), '<f4', (4, 2)),
                          (('transformation matrix', 'bbox_tform'), '<f4', (3, 3)),
                          (('Facial Landmarks', 'landmarks'), '<f4', (5, 2)),
                          ('descriptor', '<f4', 128),
                          ('label', '<i4'),
                          ('label_str', 'S16')
                          ])

    data = dataset['data']

    for frame_id, dets in data.items():
        dets_ = np.empty(len(dets), dtype=comp_type)  # composite array to store the detections
        for det_id, det in enumerate(dets):
            dets_[det_id] = (det['bbox'],
                             det['confidence'],
                             det['bbox_aligned'],
                             det['bbox_tform'],
                             det['landmarks'],
                             det['descriptor'],
                             det['label'],
                             det['label_str'])
        # save each array in a separate dataset named after the frame number in root group
        file.create_dataset(HDF5_DETECTIONS_GROUP + '/' + str(frame_id), data=dets_)

    tubelets = dataset.get('tubelets', [])

    for tub_idx, tubelet in enumerate(tubelets):
        file.create_dataset(HDF5_TUBELETS_GROUP + '/' + str(tub_idx), data=tubelet)


    file.close()



def bbox_iou(bb1, bb2):
    """
    Compute bounding box Intersection-over-Union
    :param bb1: 4 element list with top left and bottom right corner coordinates [tl.x tl.y br.x br.y]
    :param bb2:
    :return: 0 <= iou <= 1
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def parse_args(argv):
    import argparse

    # Configure the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_filename',
                        help='Video file to process')

    parser.add_argument("-d", "--dataset-filename", metavar="FILENAME", type=str,
                        help="JSON file where to save the dataset (default: vid_filename.json)")
    parser.add_argument("-m", "--message-freq", metavar="MSG-FREQ", type=int, default=100,
                        help="Frequency of progress reporting, every MSG-FREQ frames "
                             "(default: 100)")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if not os.path.exists(args.vid_filename):
        print('Cannot find video file "{}". Please make sure path to the file is correct.'.format(args.vid_filename))
        exit(1)
    dataset_fname, dataset = import_vid(args.vid_filename, dataset_fname=args.dataset_filename, msg_freq=args.message_freq)
    dataset = track_faces(dataset)

    tub_mean_desc = tubelets_mean_descriptors(dataset)
    tub_overlap = tubelet_overlap(dataset)
    labels = _chinese_whispers(tub_mean_desc.tolist(), tub_overlap)

    names_ = []
    for ll in labels:
        names_.append(names.get_first_name(gender='female'))

    for label, tubelet_ids in enumerate(labels):
        for tubelet_id in tubelet_ids:
            tubelet = dataset['tubelets'][tubelet_id]
            for k, n in tubelet:
                dataset['data'][k][n]['label_str'] = names_[label]

    annotate_vid(dataset, out_vid_fname='blah.mp4', fps=10, limit=-1)

    save_dataset(dataset, dataset_fname)
