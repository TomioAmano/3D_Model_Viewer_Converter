# 3D_Model_Viewer_Converter
A simple 3D model viewer (.obj, .vox, .mat etc)

2022/02/20  version (work in progress)
1. load 3D surface models (*.obj, *.ply), and displays them.
2. load 3D voxel model (*.vox assuming MagicaVoxel 0.982), displays it (as LEGO surface).
3. save 3D voxel model (*.vox, *.npy)
4. re-sample voxel model, then generate new voxel model with specified grid size(16x16x16, 32x32x32, 64x64x64, 128x128x128, 256x256x256).

## Depnendancies
- [vedo](https://github.com/marcomusy/vedo) 
- [py-vox-io](https://github.com/gromgull/py-vox-io)
