import tensorflow as tf

with open('tf_record_readable.txt', 'w+') as f:
	example = tf.python_io.tf_record_iterator("out.record")[0]
	result = tf.train.Example.FromString(example)
	f.write(str(result))
		
