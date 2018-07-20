"""
    Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
import time
import cv2
import os
import sys

def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'complete', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def extract_frames_from_video():
	input_loc = sys.argv[1]
	output_loc = sys.argv[2]

	try:
		os.mkdir(output_loc)
	except OSError:
		pass


	# Log the time
	time_start = time.time()
	# Start capturing the feed
	cap = cv2.VideoCapture(input_loc)
	# Find the number of frames
	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
	print ("Number of frames: ", video_length)
	count = 0
	print ("Converting video..\n")
	# Start converting the video
	while cap.isOpened():
		# Extract the frame
		ret, frame = cap.read()
		# Write the results back to output location.
		cv2.imwrite(output_loc + "/_Image_%#05d.jpg" % (count+1), frame)
		

		if count % 1000 == 0:
			printProgressBar(count, video_length)
		
		count = count + 1

		# If there are no more frames left
		if (count > (video_length-1)):
			printProgressBar(video_length, video_length)
			print()
			# Log the time again
			time_end = time.time()
			# Release the feed
			cap.release()
			# Print stats
			print ("Done extracting frames, %d frames extracted" % count)
			print ("It took %d seconds for conversion." % (time_end-time_start))
			break

extract_frames_from_video()
