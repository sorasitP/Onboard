# -*- coding:utf-8 -*-

#sudo python main.py --n_epoch=250 --method=ours-base  --dataset=cifar100 --batch_size=128
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100

import argparse, sys
import numpy as np
from data.mask_data import Mask_Select

from resnet import ResNet101

## parser for setting program argument
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '../results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')

parser.add_argument('--dataset', type = str, help = 'mnist,minimagenet, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--seed', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--network', type=str, default="resnet101")
parser.add_argument('--transforms', type=str, default="false")

parser.add_argument('--unstabitily_batch', type=int, default=16)
args = parser.parse_args()
print (args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

## building model architecture
network_map={'resnet101':ResNet101}
CNN=network_map[args.network]

## set transform functions for using in dataset
transforms_map32 = {"true": transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()]), 'false': transforms.Compose([transforms.ToTensor()])}
transformer = transforms_map32[args.transforms]

## select dataset
if args.dataset=='cifar10':
	input_channel=3
	num_classes=10
	args.top_bn = False
	args.epoch_decay_start = 80
	train_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
					noise_rate=args.noise_rate
					)
if args.dataset=='cifar100':
	input_channel=3
	num_classes=100
	args.top_bn = False
	args.epoch_decay_start = 100
	train_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)
	
## set fortget rate (to define top n% rank noise to cleanse)
if args.forget_rate is None:
	forget_rate=args.noise_rate
else:
	forget_rate=args.forget_rate

## noise_or_not array to evaluate the answer of noise detection model
noise_or_not = train_dataset.noise_or_not

## function for selecting learning rate according to maximum epoch
def adjust_learning_rate(optimizer, epoch,max_epoch=200):
	if epoch < 0.25 * max_epoch:
		lr = 0.01
	elif epoch < 0.5 * max_epoch:
		lr = 0.005
	else:
		lr = 0.001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

## function for evaluting model with test dataset
def evaluate(test_loader, model1):
	model1.eval()
	correct1 = 0
	total1 = 0
	for images, labels, _ in test_loader:
		images = Variable(images).cuda()
		#print images.shape
		logits1 = model1(images)
		outputs1 = F.log_softmax(logits1, dim=1)
		_, pred1 = torch.max(outputs1.data, 1)
		total1 += labels.size(0)
		correct1 += (pred1.cpu() == labels).sum()
	acc1 = 100 * float(correct1) / float(total1)
	model1.train()

	return acc1

## function for first_stage and third_stage training
def first_stage(network,test_loader,filter_mask=None):

	### check mast for selecting first stage or third stage
	if filter_mask is not None:#third stage
		## load dataset with cleanse noisy label by using filter_mask 
		train_loader_init = torch.utils.data.DataLoader(dataset=Mask_Select(train_dataset,filter_mask),
													batch_size=128,
													num_workers=6,
													shuffle=True,pin_memory=True)
	else:
		train_loader_init = torch.utils.data.DataLoader(dataset=train_dataset,
														batch_size=128,
														num_workers=6,
														shuffle=True, pin_memory=True)
		
	## Checkpoint saving path
	save_checkpoint=args.network+'_'+args.dataset+'_'+args.noise_type+str(args.noise_rate)+'.pt'

	## If third stage load model which saved in first stage
	if  filter_mask is not None:
		print ("restore model from %s.pt"%save_checkpoint)
		network.load_state_dict(torch.load(save_checkpoint))
	
	## number of data in dataset
	ndata=train_dataset.__len__()

	## Set optimizer and criterion
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) ## Optimizer use SGD with momentum
	criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda() # Criterion use CrossEntropyLoss

	## loop for each epochs
	for epoch in range(1, args.n_epoch):
		# train models

		## initialize global loss
		globals_loss = 0

		## set model to train Mode
		network.train()

		## setting model to not calculate gradient for evaluating model with test dataset
		with torch.no_grad():
			accuracy = evaluate(test_loader, network)

		## initial loss
		example_loss = np.zeros_like(noise_or_not, dtype=float)

		## initial learning rate
		lr=adjust_learning_rate(optimizer1,epoch,args.n_epoch)

		## Loop for training
		for i, (images, labels, indexes) in enumerate(train_loader_init):
			## convert images and labels to Cuda
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			## predict classes using images
			logits = network(images)
			## compute loss based on model output and real labels
			loss_1 = criterion(logits, labels)

			## save loss from each instances
			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			## compute global loss (sum loss of every instance)
			globals_loss += loss_1.sum().cpu().data.item()
			## average the loss
			loss_1 = loss_1.mean()
			## zero the parameter gradients
			optimizer1.zero_grad()
			## backpropagate the loss
			loss_1.backward()
			## adjust weights based on calculated gradients
			optimizer1.step()
		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuarcy:%f" % accuracy)

		## if first stage it will save
		if filter_mask is None:
			torch.save(network.state_dict(), save_checkpoint)

## function for second_stage training
def second_stage(network,test_loader,max_epoch=250):
	
	#initial variables
	train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=16,
											   num_workers=6,
											   shuffle=True)
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	moving_loss_dic=np.zeros_like(noise_or_not)
	ndata = train_dataset.__len__()


	for epoch in range(1, max_epoch):
		# train models
		globals_loss=0

		## set model to train Mode
		network.train()

		## compute accuracy
		with torch.no_grad():
			accuracy=evaluate(test_loader, network)

		## loss array for each instance (set zero every epoch)
		example_loss= np.zeros_like(noise_or_not,dtype=float)

		## calculate lr in linear equation from 0.0091 to 0.001 and delta == 0.0009
		t = (epoch % 10 + 1) / float(10)
		lr = (1 - t) * 0.01 + t * 0.001

		## change learning rate of the optimizer
		for param_group in optimizer1.param_groups:
			param_group['lr'] = lr

		## train model
		for i, (images, labels, indexes) in enumerate(train_loader_detection):

			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 = criterion(logits,labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()

			loss_1 = loss_1.mean()
			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()
		
		## normalize loss from each instance
		example_loss=example_loss - example_loss.mean()

		## update epoch loss to total loss
		moving_loss_dic=moving_loss_dic+example_loss

		## return indices of sorted array
		ind_1_sorted = np.argsort(moving_loss_dic)
		## sort total by use sorted indices array
		loss_1_sorted = moving_loss_dic[ind_1_sorted]

		## set nth rank to keep
		remember_rate = 1 - forget_rate
		num_remember = int(remember_rate * len(loss_1_sorted))

		## find accuracy of noise detection
		noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)

		## create filter_mask
		mask = np.ones_like(noise_or_not,dtype=np.float32)
		## set top n% rank to zero (for cleansing noise)
		mask[ind_1_sorted[num_remember:]]=0

		## find top 0.1 noise accuracy
		top_accuracy_rm=int(0.9 * len(loss_1_sorted))
		top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)



	return mask

## create CNN model
basenet= CNN(input_channel=input_channel, n_outputs=num_classes).cuda()

## load test dataset for evaluate model
test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,batch_size=128,
	num_workers=6,shuffle=False, pin_memory=True)

## first stage: standard model training with constant learning rate
first_stage(network=basenet,test_loader=test_loader)
## second stage: O2U-net method training with linear learning rate
filter_mask=second_stage(network=basenet,test_loader=test_loader)
## third stage: use saved model from first stage, to continue train model with cleansed dataset
first_stage(network=basenet,test_loader=test_loader,filter_mask=filter_mask)
