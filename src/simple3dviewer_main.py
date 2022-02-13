
from fileinput import filename
import sys
import os
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
    
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.cur_ix = 0
        self.frame = Qt.QFrame()
        self.layout = Qt.QHBoxLayout()
        self.grid_layout = Qt.QGridLayout()
        self.btn_layout = Qt.QVBoxLayout()

        self.mesh_list = []
        for ix in range(PARTION_NUM):
            self.mesh_list.append(None)
        self.vtkWidget_list = []
        self.vp_list = []
        self.label_list = []
        self.plotter_ix = 0

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
        button = Qt.QPushButton("Load 3D model")
        button.setFont(font)
        button.setToolTip('This is a button for loading 3D model file.')
        button.clicked.connect(self.onClickLoadButton)

        self.combobox = Qt.QComboBox()
        self.combobox.setFont(font)
        self.combobox.addItems(["1","2","3","4","5","6"])
        self.combobox.activated.connect(self.onClickSelectButton)
        self.btn_layout.addWidget(self.combobox,alignment=QtCore.AlignTop)

        self.btn_layout.addWidget(button,alignment=QtCore.AlignTop)
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

    @Qt.pyqtSlot()
    def onClickSelectButton(self):
        #printc("Kita!")
        new_ix = self.combobox.currentIndex()
        cur_ix = self.cur_ix
        cur_label = self.label_list[cur_ix]
        new_label = self.label_list[new_ix]
        new_label.setStyleSheet('background-color: black; color: white;')
        cur_label.setStyleSheet('background-color: white; color: black;')
        self.cur_ix = new_ix

    @Qt.pyqtSlot()
    def onClickLoadButton(self):
        printc("..calling onClick button \" Load 3D model \"")
        self.plotter_ix = self.combobox.currentIndex()
        #printc("current index:"+ str(cur_ix))
        file_path = self.openFileNameDialog()
        #printc(file_name)
        if file_path:
            file_name = os.path.basename(file_path)
            ix = self.plotter_ix
            
            if self.mesh_list[ix] is not None:
                self.vp_list[ix].clear()
            if file_name.endswith("npy"):
                npy_data = np.load(file_path)
                vol_data = Volume(npy_data)
                printc(type(vol_data))
                img_data = vol_data.imagedata()
                npy_data2 = numpy_support.vtk_to_numpy(img_data.GetPointData().GetScalars())
                npy_data3 = npy_data2.reshape(img_data.GetDimensions()).transpose(2,1,0)
                vol_data2 = Volume(npy_data3)

                #self.mesh_list[ix] = vol_data2.legosurface(0.9,1.1).c("grey").lighting("plastic").rotateX(180).rotateZ(180).rotateY(-90)
                self.mesh_list[ix] = vol_data2.isosurface(0.9).c("grey").lighting("plastic").rotateX(180).rotateZ(180).rotateY(-90)
                self.vp_list[ix] += self.mesh_list[ix]
            elif file_name.endswith("vox"):
                parser = VoxParser(file_path)
                m1 = parser.parse()
                size = m1.models[0].size
                voxels = m1.models[0].voxels

                EMPTY = 0
                FILLED = 1
                voxel_data = np.full((size.x, size.y, size.z), EMPTY, dtype=np.int8)
                for voxel in voxels:
                    voxel_data[voxel.x, voxel.y, voxel.z] = FILLED
                vol_data = Volume(voxel_data)
                mesh = vol_data.legosurface(0.9,1.1).c("grey").lighting("plastic")
                self.mesh_list[ix] = mesh
                self.vp_list[ix] += self.mesh_list[ix]

            else:
                mesh1 =  Mesh(file_path).c("grey").lighting("plastic").rotateX(180).rotateZ(180)
                vol1 = mesh2Volume(mesh1,spacing=(0.01, 0.01, 0.01))
                printc(type(vol1))
                mesh2 = vol1.legosurface(0.1,256.0).c("grey").lighting("plastic")
                self.mesh_list[ix] = mesh2
                self.vp_list[ix] += self.mesh_list[ix]
            self.label_list[ix].setText("Slot {0} :{1}".format(str(ix+1), file_name))
            self.vp_list[ix].show()


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('Simple 3D Viewer')
    #app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
