
from fileinput import filename
import sys
import os
import shape as sh
from PyQt5 import Qt
from PyQt5.QtCore import Qt as QtCore

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Cone, printc
from vedo import *
import numpy as np
from vtk.util import numpy_support
from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser

PARTION_NUM = 6
PARTION_PER_ROW = 3
class MainWindow(Qt.QMainWindow):

    def create_button(self, name, tip):
        button = Qt.QPushButton(name)
        button.setFont(self.font)
        button.setToolTip(tip)
        return button       

    def create_label(self, style, text):       
        label = Qt.QLabel()
        label.setFrameStyle(style)
        label.setFont(self.font)
        label.setText(text)
        return label

    def create_combobox(self,item_list):
        combobox = Qt.QComboBox()
        combobox.setFont(self.font)
        combobox.addItems(item_list)
        return combobox

    
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.cur_ix = 0
        self.frame = Qt.QFrame()
        self.layout = Qt.QHBoxLayout()
        self.grid_layout = Qt.QGridLayout()
        self.btn_layout = Qt.QVBoxLayout()
        self.shape_list = []
        self.mesh_list = []
        for ix in range(PARTION_NUM):
            self.mesh_list.append(None)
            self.shape_list.append(None)
        self.vtkWidget_list = []
        self.vp_list = []
        self.label_list = []
        self.plotter_ix = 0
        self.font = Qt.QFont("Arial",10)

        for ix in range(PARTION_NUM):
            vtk_widget = QVTKRenderWindowInteractor(self.frame)
            self.vtkWidget_list.append(vtk_widget)  
            self.vp_list.append(Plotter(qtWidget=vtk_widget))
            #self.vp_list[ix] += Mesh("9478_vox.ply").c("grey").lighting("plastic").rotateX(180).rotateZ(180)      
            self.vp_list[ix].show()
            label = Qt.QLabel()
            label.setFrameStyle(Qt.QFrame.Box | Qt.QFrame.Plain)
            font = Qt.QFont("Arial",10)
            label.setFont(font)
            label.setText("Slot {0} :Empty".format(str(ix+1)))
            self.label_list.append(label)
            partion = Qt.QVBoxLayout()
            partion.addWidget(label)
            partion.addWidget(vtk_widget)
            #self.grid_layout.addWidget(vtk_widget, ix//PARTION_PER_ROW, ix%PARTION_PER_ROW)
            self.grid_layout.addLayout(partion, ix//PARTION_PER_ROW, ix%PARTION_PER_ROW)
            #self.id1 = self.vp.addCallback("mouse click", self.onMouseClick)
            #self.id2 = self.vp.addCallback("key press",   self.onKeypress)

        self.label_list[0].setStyleSheet('background-color: black; color: white;')    
        # Set-up the rest of the Qt window

        load_label = self.create_label(Qt.QFrame.Plain,"Slot No. to Load")


        load_button = self.create_button("Load 3D model", 'This is a button for loading 3D model file.')
        load_button.clicked.connect(self.onClickLoadButton)


        load_slot_combobox = self.create_combobox(["1","2","3","4","5","6"])
        load_slot_combobox.activated.connect(self.onClickSelectButton)
        self.load_slot_combobox = load_slot_combobox

        save_label = self.create_label(Qt.QFrame.Plain,"\nSlot No. to Save")
        save_button = self.create_button("Save 3D model", 'This is a button for saving 3D model file.')
        save_button.clicked.connect(self.onClickSaveButton)
        save_slot_combobox = self.create_combobox(["1","2","3","4","5","6"])
        #save_slot_combobox.activated.connect(self.onClickSelectButton)
        self.save_slot_combobox = save_slot_combobox

        conv_label = self.create_label(Qt.QFrame.Plain,"\nSlot No. to re-sample")
        conv_button = self.create_button("Re-sample 3D model", 'This is a button for re-sampleing 3D model file.')
        conv_button.clicked.connect(self.onClickResampleButton)
        conv_slot_combobox = self.create_combobox(["1","2","3","4","5","6"])
        resample_combobox = self.create_combobox(["16x16x16","32x32x32","64x64x64","128x128x128","256x256x256"])
        self.conv_slot_combobox = conv_slot_combobox
        self.resample_combobox = resample_combobox
        self.resample_list = [16,32,64,128,256]

        dummy_label = self.create_label(Qt.QFrame.Plain,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        self.btn_layout.addWidget(load_label,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(load_slot_combobox,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(load_button,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(save_label,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(save_slot_combobox,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(save_button,alignment=QtCore.AlignTop)

        self.btn_layout.addWidget(conv_label,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(conv_slot_combobox,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(resample_combobox,alignment=QtCore.AlignTop)
        self.btn_layout.addWidget(conv_button,alignment=QtCore.AlignTop)

        self.btn_layout.addWidget(dummy_label,alignment=QtCore.AlignTop)

        self.layout.addLayout(self.btn_layout)
        self.layout.addLayout(self.grid_layout)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()                     # <--- show the Qt Window

    def openFileNameDialog(self):
        options = Qt.QFileDialog.Options()
        options |= Qt.QFileDialog.DontUseNativeDialog
        file_path, _ = Qt.QFileDialog.getOpenFileName(self,"Select 3D Model File ", "","All Files (*);;Python Files (*.py)", options=options)
        return file_path

    def saveFileNameDialog(self):
        options = Qt.QFileDialog.Options()
        options |= Qt.QFileDialog.DontUseNativeDialog
        file_path, _ = Qt.QFileDialog.getSaveFileName(self,"Input 3D Model File name ", "","All Files (*);;Python Files (*.py)", options=options)
        return file_path


    @Qt.pyqtSlot()
    def onClickSelectButton(self):
        #printc("Kita!")
        new_ix = self.load_slot_combobox.currentIndex()
        cur_ix = self.cur_ix
        cur_label = self.label_list[cur_ix]
        new_label = self.label_list[new_ix]
        new_label.setStyleSheet('background-color: black; color: white;')
        cur_label.setStyleSheet('background-color: white; color: black;')
        self.cur_ix = new_ix

    @Qt.pyqtSlot()
    def onClickLoadButton(self):
        printc("..calling onClick button \" Load 3D model \"")
        self.plotter_ix = self.load_slot_combobox.currentIndex()
        #printc("current index:"+ str(cur_ix))
        file_path = self.openFileNameDialog()
        #printc(file_name)
        if file_path:
            file_name = os.path.basename(file_path)
            ix = self.plotter_ix
            
            if self.mesh_list[ix] is not None:
                self.vp_list[ix].clear()
            if file_name.endswith("npy"):
                npy_shape = sh.Shape()
                npy_shape.read_npy(file_path)
                self.shape_list[ix] = npy_shape

                #self.mesh_list[ix] = vol_data2.legosurface(0.9,1.1).c("grey").lighting("plastic").rotateX(180).rotateZ(180).rotateY(-90)
                self.mesh_list[ix] = npy_shape.vedo_legomesh #vol_data2.isosurface(0.9).c("grey").lighting("plastic").rotateX(180).rotateZ(180).rotateY(-90)
                self.vp_list[ix] += self.mesh_list[ix]
            elif file_name.endswith("vox"):

                vox_shape = sh.Shape()
                vox_shape.read_vox(file_path)
                self.shape_list[ix] = vox_shape

                self.mesh_list[ix] = vox_shape.vedo_legomesh
                self.vp_list[ix] += self.mesh_list[ix]

            elif file_name.endswith("mat"):
                vox_shape = sh.Shape()
                vox_shape.read_mat(file_path)
                self.shape_list[ix] = vox_shape

                self.mesh_list[ix] = vox_shape.vedo_legomesh
                self.vp_list[ix] += self.mesh_list[ix]               
            else:
                mesh_shape = sh.Shape()
                mesh_shape.read_mesh(file_path)
                self.shape_list[ix] = mesh_shape

                self.mesh_list[ix] = mesh_shape.vedo_mesh.c("grey").lighting("plastic")
                self.vp_list[ix] += self.mesh_list[ix]
            self.label_list[ix].setText("Slot {0} :{1}".format(str(ix+1), file_name))
            self.vp_list[ix].show()

    @Qt.pyqtSlot()   
    def onClickResampleButton(self):
        plotter_ix = self.conv_slot_combobox.currentIndex()
        resample_ix = self.resample_combobox.currentIndex()
        vox_size = self.resample_list[resample_ix]
        src_shape = self.shape_list[plotter_ix]
        if src_shape.model_type == src_shape.VOXEL_MODEL:
            dst_shape = src_shape.voxel2voxel(vox_size)
        elif src_shape.model_type == src_shape.SURFACE_MODEL:
            dst_shape = src_shape.mesh2voxel(vox_size)
        self.shape_list[plotter_ix] = dst_shape
        self.mesh_list[plotter_ix] = dst_shape.vedo_legomesh.c("grey").lighting("plastic")
        self.vp_list[plotter_ix].clear()
        self.vp_list[plotter_ix] += self.mesh_list[plotter_ix]
        self.vp_list[plotter_ix].show()

    def save_shape(self, shape, file_path):
        if file_path.endswith("obj"):
            if shape.model_type == shape.SURFACE_MODEL:
                shape.save_obj(file_path)
            elif shape.model_type == shape.VOXEL_MODEL:
                mesh = shape.voxel2mesh()
                mesh.save_obj(file_path)
            return
        if shape.model_type == shape.VOXEL_MODEL:
            if file_path.endswith("npy"):
                shape.save_npy(file_path)
            elif file_path.endswith("vox"):
                shape.save_vox(file_path)

    @Qt.pyqtSlot()
    def onClickSaveButton(self):
        slot_ix = self.save_slot_combobox.currentIndex()
        file_path = self.saveFileNameDialog()
        shape = self.shape_list[slot_ix]
        if file_path:
            self.save_shape(shape, file_path)

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('Simple 3D Viewer')
    #app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
