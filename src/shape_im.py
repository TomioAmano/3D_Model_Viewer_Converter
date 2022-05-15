import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import torch
import mcubes
from utils import *

from modelAE import IM_AE
import argparse

class IM_Morph(IM_AE):
	def __init__(self, flags):
		super().__init__(flags)

		#output shape as ply and point cloud as ply
	def perpare_eval(self):
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		return
	
	def load_checkpoint(self):
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

	def decode_zdim(self, zdim):
		model_z = torch.Tensor(zdim.reshape(1,256)).cuda()
		model_float = self.z2voxel(model_z)
		vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
		vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
		write_ply_triangle("temp_vox.ply", vertices, triangles)
		return model_float

	def get_zdim(self, voxel):
		dim_size = max(voxel.shape)
		batch_voxels_ = voxel.reshape(1,1,dim_size, dim_size, dim_size).astype(np.float32)
		max_val = batch_voxels_.max()
		min_val = batch_voxels_.min()
		#np.save("temp.npy", batch_voxels_.reshape(64,64,64))
		# 座標系の違いで横向きになっているのを補正する
		batch_voxels_ = batch_voxels_.transpose((0,1,4,3,2))
		batch_voxels = torch.from_numpy(batch_voxels_)
		batch_voxels = batch_voxels.to(self.device)
		model_z,_ = self.im_network(batch_voxels, None, None, is_training=False)
		return model_z

	def get_voxel(self, model_z):
		model_float = self.z2voxel(model_z)
		return model_float

	def test_mesh_point_new(self, config):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		for t in range(config.start, min(len(self.data_voxels),config.end)):
			batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
			np.save("temp.npy", batch_voxels_.reshape(64,64,64))
			batch_voxels = torch.from_numpy(batch_voxels_)
			batch_voxels = batch_voxels.to(self.device)
			model_z,_ = self.im_network(batch_voxels, None, None, is_training=False)
			model_float = self.z2voxel(model_z)

			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply_triangle(config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)
			
			print("[sample]")
			
			#sample surface points
			sampled_points_normals = sample_points_triangle(vertices, triangles, 4096)
			np.random.shuffle(sampled_points_normals)
			write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals)
			
			print("[sample]")

	

class Shape_IM:
	def __init__(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
		parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
		parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
		parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
		parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
		parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
		parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
		parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
		parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
		parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
		parser.add_argument("--start", action="store", dest="start", default=8400, type=int, help="In testing, output shapes [start:end]")
		parser.add_argument("--end", action="store", dest="end", default=8450, type=int, help="In testing, output shapes [start:end]")
		parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
		parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
		parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
		FLAGS = parser.parse_args()

		if not os.path.exists(FLAGS.sample_dir):
			os.makedirs(FLAGS.sample_dir)

		self.im_ae = IM_Morph(FLAGS)


		"""
		if FLAGS.train:
			im_ae.train(FLAGS)
		elif FLAGS.getz:
			im_ae.get_z(FLAGS)
		else:
			#im_ae.test_mesh(FLAGS)
			im_ae.test_mesh_point_new(FLAGS)
		"""

if __name__ == "__main__":
	shape = Shape_IM()
	print("Kita!")

