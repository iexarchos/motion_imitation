#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:04:49 2020

@author: yannis
"""


import pybullet as p
import pybullet_data
import numpy as np
import os
from pdb import set_trace as bp

scales = [0.8, 0.5 , 0.4, 0.9]

#front pair of legs 
front_upper = scales[0]*0.2
front_lower = scales[1]*0.2
back_upper = scales[2]*0.2
back_lower = scales[3]*0.2
#link offsets
front_upper_off = -front_upper*0.5
front_lower_off = -front_lower*0.5
back_upper_off = -back_upper*0.5
back_lower_off = -back_lower*0.5


FU = "{0:.6f}".format(front_upper)
FL = "{0:.6f}".format(front_lower)
BU = "{0:.6f}".format(back_upper)
BL = "{0:.6f}".format(back_lower)
OFU = "{0:.6f}".format(front_upper_off)
OFL = "{0:.6f}".format(front_lower_off)
OBU = "{0:.6f}".format(back_upper_off)
OBL = "{0:.6f}".format(back_lower_off)



proc_ID = str(os.getpid())
exec_dir = os.getcwd() #<-- absolute dir the script is in
#abs_file_path = self.exec_dir+'/my_pybullet_envs/assets/laikago/laikago_SCALE.urdf'
abs_file_path = exec_dir+'/motion_imitation/robots/a1_template.urdf'
with open(abs_file_path, 'r') as file :
    filedata = file.read()
    filedata = filedata.replace('FrUp', FU)
    filedata = filedata.replace('FrLo', FL)
    filedata = filedata.replace('BaUp', BU)
    filedata = filedata.replace('BaLo', BL)
    filedata = filedata.replace('OfsFrU', OFU)
    filedata = filedata.replace('OfsFrL', OFL)
    filedata = filedata.replace('OfsBaU', OBU)
    filedata = filedata.replace('OfsBaL', OBL)


with open(exec_dir+'/motion_imitation/robots/temp_URDFs/'+proc_ID+'.urdf', 'w') as file:
            file.write(filedata)

p.connect(p.GUI)
p.setPhysicsEngineParameter(enableFileCaching=0) # IMPORTANT! This is needed to avoid using the same cached urdf file every time!
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#robot = p.loadURDF('/home/yannis/Dropbox/Research_Projects/co-design-control/grasping/temp_URDFs/1.urdf',useFixedBase=1)          
robot = p.loadURDF(exec_dir+'/motion_imitation/robots/temp_URDFs/'+proc_ID+'.urdf')
bp()


#robot2 = p.loadURDF('a1/a1.urdf')
