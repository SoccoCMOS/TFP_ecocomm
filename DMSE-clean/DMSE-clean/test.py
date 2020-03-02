import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model
import get_data 
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math
import urllib
from pyheatmap.heatmap import HeatMap
import seaborn as sns

FLAGS = tf.app.flags.FLAGS

def analysis(species, indiv_prob, input_label, printNow = False):
	

	TP = 0
	TN = 0
	FN = 0
	FP = 0
	pred_label = np.greater(indiv_prob, FLAGS.threshold).astype(int)

	#print input_label.shape 
	for j in range(input_label.shape[1]):
		for i in range(input_label.shape[0]):
			if (pred_label[i][j]==1 and input_label[i][j]==1):
				TP+=1
			if (pred_label[i][j]==1 and input_label[i][j]==0):
				FP+=1
			if (pred_label[i][j]==0 and input_label[i][j]==0):
				TN+=1
			if (pred_label[i][j]==0 and input_label[i][j]==1):
				FN+=1

	N = (TP+TN+FN+FP)*1.0
	eps = 1e-6
	precision = 1.0 * TP / (TP + FP + eps)
	recall = 1.0 * TP / (TP + FN + eps) 
	Accuracy =  1.0*(TN+TP)/(N)
	F1 = 2.0 * precision * recall / (precision + recall + eps)
	#print "F2:", (1+4)*precision*recall/(4*precision+recall)

	occurrence = np.mean(input_label)
	auc = roc_auc_score(input_label, indiv_prob)

	indiv_prob = np.reshape(indiv_prob, (-1))
	input_label = np.reshape(input_label, (-1))

	new_auc = roc_auc_score(input_label, indiv_prob)

	ap = average_precision_score(input_label,indiv_prob)

	if (printNow):
		print "\nThis is the analysis of #%s species:"%species
		print "occurrence rate:", occurrence
		print "Overall \tauc=%.6f\tnew_auc=%.6f\tap=%.6f" % (auc, new_auc, ap)	
		print "F1:", F1 
		print "Accuracy:", Accuracy
		print "Precision:", precision
		print "Recall:", recall
		print "TP=%f, TN=%f, FN=%f, FP=%f"%(TP/N, TN/N, FN/N, FP/N)
		print " "
	return occurrence, auc, F1, Accuracy, new_auc, ap,  precision, recall, TP/N, TN/N, FN/N, FP/N,

def main(_):

	print 'reading npy...'

	data = np.load(FLAGS.data_dir)
	test_idx = np.load(FLAGS.test_idx)

	print 'reading completed'

	session_config = tf.ConfigProto()
	session_config.gpu_options.allow_growth = True
	sess = tf.Session(config=session_config)

	print 'building network...'

	classifier = model.MODEL(is_training=False)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)

	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():
		print 'Testing...'
		all_nll_loss = 0
		all_l2_loss = 0
		all_total_loss = 0

		all_indiv_prob = []
		all_label = []

		sigma=[]
		real_batch_size=min(FLAGS.testing_size, len(test_idx))
		avg_cor=0
		cnt_cor=0
		
		N_test_batch = int( (len(test_idx)-1)/real_batch_size )+1
		#print N_test_batch

		for i in range(N_test_batch):

			print "%.1f%% completed" % (i*100.0/N_test_batch)

			start = real_batch_size*i
			end = min(real_batch_size*(i+1), len(test_idx))

			#input_image= get_data.get_image(images, test_idx[start:end])
			input_nlcd = get_data.get_nlcd(data,test_idx[start:end])
			input_label = get_data.get_label(data,test_idx[start:end])

			feed_dict={}
			feed_dict[classifier.input_nlcd]=input_nlcd
			feed_dict[classifier.input_label]=input_label
			#feed_dict[classifier.input_image]=input_image
			feed_dict[classifier.keep_prob]=1.0
			

			nll_loss, l2_loss, total_loss, indiv_prob, covariance= sess.run([classifier.nll_loss, classifier.l2_loss, \
				classifier.total_loss, classifier.indiv_prob, classifier.covariance],feed_dict)
			
			all_nll_loss += nll_loss*(end-start)
			all_l2_loss += l2_loss*(end-start)
			all_total_loss += total_loss*(end-start)

			if (all_indiv_prob == []):
				all_indiv_prob = indiv_prob
			else:
				all_indiv_prob = np.concatenate((all_indiv_prob, indiv_prob))

			if (all_label == []):
				all_label = input_label
			else:
				all_label = np.concatenate((all_label, input_label))

		
		#print "Overall occurrence ratio: %f"%(np.mean(all_label))
		
		nll_loss = all_nll_loss / len(test_idx)
		l2_loss = all_l2_loss / len(test_idx)
		total_loss = all_total_loss / len(test_idx)

		time_str = datetime.datetime.now().isoformat()

		print "performance on test_set: nll_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f \n%s" % (nll_loss, l2_loss, total_loss, time_str)	
		
		#print FLAGS.visual_dir+"cov"
		np.save(FLAGS.visual_dir+"cov", covariance)
		return all_indiv_prob, all_label

		


	indiv_prob, input_label = test_step()

	"""analysis("all", indiv_prob, input_label, True)

	summary = []
	for i in range(FLAGS.r_dim):
		#print i
		sp_indiv_prob = indiv_prob[:,i].reshape(indiv_prob.shape[0],1)
		sp_input_label = input_label[:,i].reshape(input_label.shape[0],1)

		res = analysis(i, sp_indiv_prob, sp_input_label, False)
		summary.append(res)
	summary = np.asarray(summary)
	

	np.save("../data/summary_all",summary) #save the analysis result to ../data/summary_all.npy, which loged the performance indicators of each entity
	"""
	
if __name__=='__main__':
	tf.app.run()



