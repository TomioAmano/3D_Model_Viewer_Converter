from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser
from vedo import *
import numpy as np
import scipy.io
from shape_im import Shape_IM

GRID_SCALE = 100
UNKNOWN = -1
BACKGROUND = 2
EMPTY = 0
FILLED = 1

class Triangle:
    def city_block_distance(self, point1, point2):
        return int(abs(point1[0]-point2[0])+abs(point1[1]-point2[1])+abs(point1[2]-point2[2]))

    def __init__(self, p1, p2, p3):
        self.p1= p1
        self.p2 = p2
        self.p3 = p3
        self.edge12 = self.city_block_distance(p1,p2)
        self.edge23 = self.city_block_distance(p2,p3)
        self.edge31 = self.city_block_distance(p3,p1)
    
    def divide(self):
        if self.edge12 >= max(self.edge23,self.edge31):
            np = [(self.p1[ix]+self.p2[ix])//2 for ix in range(3)]
            tri1 = Triangle(self.p3,np, self.p1)
            tri2 = Triangle(self.p3,np, self.p2)
        elif self.edge23 >= max(self.edge12,self.edge31):
            np = [(self.p2[ix]+self.p3[ix])//2 for ix in range(3)]
            tri1 = Triangle(self.p1,np, self.p2)
            tri2 = Triangle(self.p1,np, self.p3)
        else:
            np = [(self.p1[ix]+self.p3[ix])//2 for ix in range(3)]
            tri1 = Triangle(self.p2,np, self.p1)
            tri2 = Triangle(self.p2,np, self.p3)
        return (tri1, tri2)

    def get_grid(self, point, unit, dim):
        thr = dim - 1
        result =  [min(point[0]//unit,thr), min(point[1]//unit,thr), min(point[2]//unit,thr)]
        #print("===============================")
        #print("point",point)
        #print("result",result)
        return result

    def is_enough_divided(self, unit):
        max_dist = max([self.city_block_distance(self.p1, self.p2),
                        self.city_block_distance(self.p1, self.p3),
                        self.city_block_distance(self.p2, self.p3),
                        ])
        return max_dist < unit

class Shape() :
    def __init__(self):
        self.vertex_list = None
        self.face_list = None
        self.npy_voxels = None
        self.vedo_volume = None
        self.vedo_mesh = None
        self.vedo_legomesh = None
        self.voxel_size = None
        self.zdim = None
        self.SURFACE_MODEL = 1
        self.VOXEL_MODEL = 2
        self.shape_im = Shape_IM()
        self.shape_ae = self.shape_im.im_ae
        self.shape_ae.load_checkpoint()

    # z256次元データの読み込み    
    def read_z256(self, file_path):
        self.model_type = self.VOXEL_MODEL
        np_z256 = np.load(file_path)
        self.zdim = np_z256
        npy_voxels = self.shape_ae.decode_zdim(np_z256)

        self.voxel_size = 64
        EMPTY = 0
        FILLED = 1
        voxel_data = np.full((self.voxel_size, self.voxel_size, self.voxel_size), EMPTY, dtype=np.int8)
        grid = [(ix,iy,iz) for ix in range(64) for iy in range(64) for iz in range(64)]
        partion = [(dx,dy,dz) for dx in range(4) for dy in range(4) for dz in range(4)]

        for ix,iy,iz in grid:
            for dx,dy,dz in partion:
                point_val = npy_voxels[ix*4+dx+1, iy*4+dy+1, iz*4+dz+1]
                if point_val > 0.5:
                    voxel_data[ix,iy,iz] = FILLED
                    break
        voxel_data_ = voxel_data.transpose([2,1,0])                        
        self.npy_voxels = voxel_data_
        vol_data = Volume(voxel_data_)
        self.vedo_legomesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")


    def read_mesh(self, file_path):
        self.model_type = self.SURFACE_MODEL
        mesh = Mesh(file_path)
        self.vedo_mesh = mesh
        self.vertex_list = mesh.points()
        self.face_list = mesh.faces()

    def read_npy(self, file_path):
        self.model_type = self.VOXEL_MODEL
        npy_voxels = np.load(file_path)
        self.voxel_size = max(npy_voxels.shape[0],npy_voxels.shape[1],npy_voxels.shape[2])
        self.npy_voxels = npy_voxels.reshape([self.voxel_size,self.voxel_size,self.voxel_size])
        vol_data = Volume(self.npy_voxels)
        self.vedo_mesh = vol_data.isosurface(0.9)
        self.vedo_legomesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")
    
    def read_vox(self, file_path):
        self.model_type = self.VOXEL_MODEL
        parser = VoxParser(file_path)
        m1 = parser.parse()
        size = m1.models[0].size
        voxels = m1.models[0].voxels
        self.voxel_size = max(size.x, size.y, size.z)

        EMPTY = 0
        FILLED = 1
        voxel_data = np.full((self.voxel_size, self.voxel_size, self.voxel_size), EMPTY, dtype=np.int8)
        for voxel in voxels:
            voxel_data[self.voxel_size - voxel.y - 1, voxel.z,voxel.x ] = FILLED
        self.npy_voxels = voxel_data
        vol_data = Volume(voxel_data)
        self.vedo_legomesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")

    def read_mat(self, file_path):
        self.model_type = self.VOXEL_MODEL
        self.voxel_size = 256
        voxel_model_mat = scipy.io.loadmat(file_path)
        data_b = voxel_model_mat["b"]
        data_bi = voxel_model_mat["bi"]
        voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
        voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
        voxel_model_256 = np.zeros([256,256,256],np.uint8)
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
        voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)
        voxel_model_256_a = voxel_model_256.reshape([1,voxel_model_256.shape[0],voxel_model_256.shape[1],voxel_model_256.shape[2],1])
        self.npy_voxels = voxel_model_256_a.reshape([256,256,256])
        vol_data = Volume(self.npy_voxels)
        self.vedo_mesh = vol_data.isosurface(0.9)
        self.vedo_legomesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")

    def normalize_vertecies(self, v_list, dim = 16, unit= GRID_SCALE):
        # size normaliztion
        min_list = [10000.0, 10000.0, 10000.0]
        max_list = [-10000.0, -10000.0, -10000.0]
        for vertex in v_list:
            min_list = [min(min_list[0], vertex[0]), min(min_list[1], vertex[1]),min(min_list[2], vertex[2])]
            max_list = [max(max_list[0], vertex[0]), max(max_list[1], vertex[1]),max(max_list[2], vertex[2])]
        size_by_axis = [max_list[ix]-min_list[ix] for ix in range(3)]
        max_size = max(size_by_axis)
        min_pos = min(min_list)
        #v_list = [[vertex[0]+max_size/2, vertex[1]+max_size/2,vertex[2]+max_size/2] for vertex in v_list]
        v_list = [[vertex[0]-min_pos, vertex[1]-min_pos,vertex[2]-min_pos] for vertex in v_list]
        scale = ((dim-2) * unit) / max_size
        # print("scale=",scale)
        iv_list = [[int(vertex[0]*scale+unit),int(vertex[1]*scale+unit),int(vertex[2]*scale+unit) ] for vertex in v_list]

        # center position normalization 要デバッグ
        min_ilist = [dim*unit, dim*unit, dim*unit]
        max_ilist = [0,0,0]
        for ivertex in iv_list:
            min_ilist = [min(min_ilist[0], ivertex[0]), min(min_ilist[1], ivertex[1]),min(min_ilist[2], ivertex[2])]
            max_ilist = [max(max_ilist[0], ivertex[0]), max(max_ilist[1], ivertex[1]),max(max_ilist[2], ivertex[2])]
        center =  [(min_v + max_v)//2 for (min_v, max_v) in zip(min_ilist, max_ilist)]
        ajust = [(dim*unit//2 - cv)//2 for cv in center ]
        #ajust = [400,0,0]
        n_iv_list = []
        for ivertex in iv_list:
            n_iv_list.append([i_val + ajust_val for (i_val, ajust_val) in zip(ivertex, ajust)])

        return n_iv_list

    def fill_grid(self,triangle, voxels):
        p1 = triangle.get_grid(triangle.p1,GRID_SCALE,voxels.shape[0])
        voxels[p1[0], p1[1], p1[2]] = FILLED
        p2 = triangle.get_grid(triangle.p2,GRID_SCALE,voxels.shape[0])
        voxels[p2[0], p2[1], p2[2]] = FILLED
        p3 = triangle.get_grid(triangle.p3,GRID_SCALE,voxels.shape[0])
        voxels[p3[0], p3[1], p3[2]] = FILLED
        if not triangle.is_enough_divided(GRID_SCALE):
            tri1, tri2 = triangle.divide()
            self.fill_grid(tri1, voxels)
            self.fill_grid(tri2, voxels)
        del triangle        

    def voxelize_mesh(self,v_list, f_list, voxels):
        iv_list = self.normalize_vertecies(v_list, dim=voxels.shape[0], unit=GRID_SCALE)
        for face in f_list:
            if len(face) == 3:
                triangle = Triangle(iv_list[face[0]], iv_list[face[1]],iv_list[face[2]])
                self.fill_grid(triangle,voxels)
            elif len(face) == 4:
                triangle1 = Triangle(iv_list[face[0]], iv_list[face[1]],iv_list[face[2]])
                triangle2 = Triangle(iv_list[face[2]], iv_list[face[3]],iv_list[face[0]])
                self.fill_grid(triangle1,voxels)
                self.fill_grid(triangle2,voxels)

    def fill_background(self, voxels):
        dim = voxels.shape[0]
        for ix in range(dim):
            for iy in range(dim):
                for iz in range(dim):
                    if voxels[ix,iy,iz] != UNKNOWN:
                        break
                    voxels[ix,iy,iz] = BACKGROUND
                for iz in range(dim):
                    riz = dim - iz -1
                    if voxels[ix,iy,riz] != UNKNOWN:
                        break
                    voxels[ix,iy,riz] = BACKGROUND
        # Propagete EMPTY Points
        while True:
            changed = False
            for ix in range(1, dim-1):
                for iy in range(1, dim-1):
                    for iz in range(1, dim-1):
                        if voxels[ix,iy,iz] != UNKNOWN:
                            continue
                        for dx in (-1, 1):
                            if voxels[ix+dx, iy, iz] == BACKGROUND:
                                voxels[ix,iy,iz] = BACKGROUND
                                changed = True
                        for dy in (-1, 1):        
                            if voxels[ix, iy+dy, iz] == BACKGROUND:
                                voxels[ix,iy,iz] = BACKGROUND
                                changed = True                            
                        for dz in (-1, 1):        
                            if voxels[ix, iy, iz+dz] == BACKGROUND:
                                voxels[ix,iy,iz] = BACKGROUND
                                changed = True 
            if not changed:
                break
        for ix in range(dim):
            for iy in range(dim):
                for iz in range(dim):
                    if voxels[ix,iy,iz] == BACKGROUND:
                        voxels[ix,iy,iz] = EMPTY
                    if voxels[ix,iy,iz] == UNKNOWN:
                        voxels[ix,iy,iz] = FILLED               

    def voxelize(self,mesh, dim=64):
        f_list = mesh.faces()
        v_list = mesh.points()
        voxels = np.full((dim, dim, dim), UNKNOWN, dtype=np.int8)
        self.voxelize_mesh(v_list,f_list,voxels)
        self.fill_background(voxels)
        # Rotate 90 degree
        voxels = np.rot90(voxels,axes=(2,0))
        return voxels

    def mesh2voxel(self, voxel_size):
        voxel_shape = Shape()
        voxel_shape.npy_voxels  = self.voxelize(self.vedo_mesh, dim=voxel_size)
        voxel_shape.model_type = self.VOXEL_MODEL
        voxel_shape.voxel_size = voxel_size
        vol_data = Volume(voxel_shape.npy_voxels)
        voxel_shape.vedo_mesh = vol_data.isosurface(0.9)
        voxel_shape.vedo_legomesh = vol_data.legosurface(0.9,1.1)
        return voxel_shape
    
    def voxel2mesh(self):
        mesh_shape = Shape()
        mesh_shape.model_type = self.SURFACE_MODEL
        vol_data = Volume(self.npy_voxels)
        mesh = vol_data.isosurface(0.9)
        mesh_shape.vedo_mesh = mesh
        mesh_shape.vertex_list = mesh.points()
        mesh_shape.face_list = mesh.faces()
    
    def voxel2voxel(self, voxel_size):
        vol_data = Volume(self.npy_voxels)
        mesh = vol_data.isosurface(0.9)
        voxel_shape = Shape()
        voxel_shape.npy_voxels  = self.voxelize(mesh, dim=voxel_size)
        voxel_shape.model_type = self.VOXEL_MODEL
        voxel_shape.voxel_size = voxel_size
        vol_data = Volume(voxel_shape.npy_voxels)
        voxel_shape.vedo_mesh = vol_data.isosurface(0.9)
        voxel_shape.vedo_legomesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")
        return voxel_shape       

    def save_vox(self, file_path):
        npy_data = self.npy_voxels > 0.8
        npy_data4voxel = np.flip(npy_data,(0,1)).copy()
        vox_data = Vox.from_dense(npy_data4voxel)
        VoxWriter(file_path, vox_data).write()
    
    def save_npy(self, file_path):
        np.save(file_path, self.npy_voxels)

    def save_obj(self,file_path):
        v_list = self.vertex_list
        f_list = self.face_list
        f = open(file_path, mode="w", encoding="utf-8")
        for vertex in v_list:
            f.write("v {:.5f} {:.5f} {:.5f}\n".format(vertex[0], vertex[1], vertex[2]))
        for face in f_list:
            if len(face) == 3:
                f.write("f {:d} {:d} {:d}\n".format(face[0]+1, face[1]+1, face[2]+1))
            elif len(face) == 4:
                f.write("f {:d} {:d} {:d} {:d}\n".format(face[0]+1, face[1]+1, face[2]+1, face[3]+1))
        f.close()

    def save_z256(self, file_path):
        zdim = self.shape_ae.get_zdim(self.npy_voxels).cpu().detach().numpy()
        np.save(file_path, zdim)

if __name__ == "__main__":
    shape1 = Shape()
    
    shape1.read_mesh("sample_data\cube.obj")
    voxel = shape1.mesh2voxel(16)
    voxel.save_vox("sample_data\cube16.vox")
    voxel2 = voxel.voxel2voxel(32)
    voxel2.save_vox("sample_data\cube32.vox")
    
    shape1.read_npy("sample_data\\robot_face_000001.npy")
    shape1.save_vox("sample_data\\robot_face_000001.vox")




