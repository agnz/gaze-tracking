import tensorflow as tf

with open('tf_record_readable.txt', 'w+') as f:
	for example in tf.python_io.tf_record_iterator("out.record"):
		result = tf.train.Example.FromString(example)
		f.write(str(result))
		break
