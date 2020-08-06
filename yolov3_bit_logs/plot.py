import matplotlib.pyplot as plt
import ast
import os

from collections import OrderedDict
dirs_dict = OrderedDict()

# dirs_dict = {
# 	'yolov3_darknet'  : 'checkpoint_from_scratch',
# 	'yolov3_resnet50' : 'checkpoint_resnet50',
# 	'yolov3_resnet101': 'checkpoint_resnet101',
# 	# 'yolov3_resnet152': 'checkpoint_resnet152',
# }

dirs_dict = {
	'yolov3_darknet'	    : 'checkpoint_from_scratch',
	'yolov3_resnet050x1_BiT': 'checkpoint_resnet50x1',
	'yolov3_resnet101x1_BiT': 'checkpoint_resnet101x1',
	'yolov3_resnet152x2_BiT': 'checkpoint_resnet152x2',
}

class Log():
	def __init__(self, log_file, type):
		self.maps 		= []
		self.lrs  		= []
		self.losses 	= []
		self.exp_type	= type
		self.logfile 	= log_file
		self.logs 		= open(self.logfile, 'r').readlines()

	def update(self, map, lr, loss):
		self.maps.append(map)
		self.lrs.append(lr)
		self.losses.append(loss)
		self.best_map 	= max(self.maps)
		self.best_loss 	= min(self.losses)


def plot_losses(logs):

	total_logs = len(logs)

	fig, axs = plt.subplots(total_logs,1)	
	for i,log in enumerate(logs):

		axs[i].set_title(log.exp_type)
		axs[i].plot(log.losses)
		
		print(log.exp_type)
		print(log.best_loss)
		print('\n')

	fig.suptitle("losses")	
	plt.show()

def plot_mAPs(logs):

	total_logs = len(logs)
	fig, axs = plt.subplots(total_logs,1)
	for i,log in enumerate(logs):		
		# axs[i].text(100, 100, "txt")
		axs[i].set_title(log.exp_type)
		axs[i].plot(log.maps)

		print(log.exp_type)
		print(log.best_map)
		print('\n')

	fig.suptitle("mAP")	
	plt.show()


logging_objects = []

for type, dir in dirs_dict.items():		

	log_file= os.path.join(dir, "log.txt")
	print(log_file)
	logging_obj = Log(log_file, type) 
	
	for line in logging_obj.logs:
		res = ast.literal_eval(line)
		logging_obj.update(res['map']*100, res['lr'], res['loss'])
	logging_objects.append(logging_obj)

plot_losses(logging_objects)
plot_mAPs(logging_objects)