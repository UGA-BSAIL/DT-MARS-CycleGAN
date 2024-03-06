import math
import random
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import re
import time
from glob import glob
import pandas as pd
import os
from scipy import ndimage
from tqdm import tqdm


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    return stats

def mask_find_multi_bboxs(mask):
    objects = ndimage.find_objects(mask)
    return objects


class World(object):
    def __init__(self, plant):
        # self.server_id = p.connect(p.GUI)
        self.server_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")
        base_position = [0,0,0.05]

        """
        change the object urdf file, choose from below

        ['big_plant', 'big_plant_src', 'cirsium', 'cirsium_v1',
        'crabgrass_v1', 'crabgrass_v2', 'polygonum_v1', 'polygonum_v2', 'small_plant']

        bbox not good: ['CottonPlant_v1', 'CottonPlant_v2', 'CottonPlant_v3']

        To adjust the plant object size, go to urdf file and change 'scale'
        """

        plant = plant
        
        # self.object_id = p.loadURDF('./weedbot_simulation/weedbot_description/urdf/weedbot.urdf', basePosition=base_position) # weedbot robot car
        # self.object_id = p.loadURDF('./weedbot_simulation/simulation_world/urdf/{}.urdf'.format(plant), basePosition=base_position) # weeds

        # multi objs
        self.object_ids = []
        for _ in range(3):
            object_id = p.loadURDF('./weedbot_simulation/simulation_world/urdf/{}.urdf'.format(plant), basePosition=base_position)
            self.object_ids.append(object_id)

    def close(self):
        p.disconnect(self.server_id)

    def change_plane(self):
        texture_list = glob('./soil_resized/*.jpg')
        texture_path = np.random.choice(texture_list)
        texture_id = p.loadTexture(texture_path)
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=texture_id)

    def step(self):
        self.change_plane()
        base_position = self.reset_object()
        rgb, seg, gt_bbox = self.get_image(base_position)
        return rgb, seg, gt_bbox

    def reset_object(self):
        anchor_position = [random.random(),random.random(),0.05]
        shift = [[-random.uniform(0.2, 0.3), -random.uniform(0.2, 0.3), 0],
                 [0, 0, 0],
                 [random.uniform(0.2, 0.3), random.uniform(0.2, 0.3), 0]]


        # p.resetBasePositionAndOrientation(self.object_id, base_position,[0,0,0,1])
        # # change color of object to green like
        # alpha = np.random.uniform(0.4, 0.8)
        # green = np.random.uniform(0.4, 0.8)
        # coneColor = [0, green, 0, alpha]
        # p.changeVisualShape(self.object_id, -1, rgbaColor=coneColor)

        # multi objs
        for i, object_id in enumerate(self.object_ids):
            base_position = [x+y for x,y in zip(anchor_position,shift[i])]
            p.resetBasePositionAndOrientation(object_id, base_position, [0, 0, 0, 1])

            # Change color of object to a random shade of green
            alpha = np.random.uniform(0.4, 0.8)
            green = np.random.uniform(0.4, 0.8)
            coneColor = [0, green, 0, alpha]
            p.changeVisualShape(object_id, -1, rgbaColor=coneColor)
        return anchor_position

    def get_image(self, base_position):
        # adjust camera hight and distance here
        r = 0.8 + 0.4*random.random() # distance
        t = 2 * math.pi * random.random()
        h = 0.8 + 0.4 * random.random() # hight
        camera_pos = [base_position[0] + r*math.sin(t), base_position[1] + r*math.cos(t), base_position[2]+h]

        target_position = [base_position[0]-0.2*random.random()+0.1,
                           base_position[1]-0.2*random.random()+0.1,
                           base_position[2]-0.1*random.random()+0.05]

        view_mat = p.computeViewMatrix(camera_pos, target_position, [0, 0, 1], self.server_id)
        proj_mat = p.computeProjectionMatrixFOV(fov=49.1,
                                                aspect=1.0,
                                                nearVal=0.1,
                                                farVal=100,
                                                physicsClientId=self.server_id)
        w, h, rgb, depth, seg = p.getCameraImage(width=640,
                                                 height=640,
                                                 viewMatrix=view_mat,
                                                 projectionMatrix=proj_mat,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb)[:, :, :3][:, :, ::-1]
        rgb = np.array(rgb, dtype=np.uint8)

        seg = np.array(seg, dtype=np.uint8)
        bboxes = mask_find_multi_bboxs(seg)
        gt_bbox = np.zeros((3,4))
        for i, b in enumerate(bboxes):
            sy, sx = b
            gt_bbox[i,0] = sx.start/w
            gt_bbox[i,1] = sy.start/h
            gt_bbox[i,2] = sx.stop/w
            gt_bbox[i,3] = sy.stop/h




        # ret, thresh = cv2.threshold(seg, 0, 255, 0)
        # bboxes = mask_find_bboxs(thresh)
        # gt_bbox = np.zeros((1,4))
        # for b in bboxes:
        #     if (b[0] + b[1]) < 100:
        #         continue
        #     gt_bbox[0,0] = b[0]/w
        #     gt_bbox[0,1] = b[1]/h
        #     gt_bbox[0,2] = b[0]/w + b[2]/w
        #     gt_bbox[0,3] = b[1]/h + b[3]/h
        return rgb,seg,gt_bbox

if __name__ == '__main__':

    plants = ['big_plant', 'small_plant', 'polygonum_v2', 'cirsium'] # 800 800 400 400
    plant = 'small_plant'

    env = World(plant=plant)
    bbox = []
    for i in tqdm(range(1200)):
        rgb, seg, gt_bbox = env.step()
        seg[seg>0] = 255
        seg3 = np.stack([seg, seg, seg], axis=2)

        color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
        thickness = 1 # Line thickness of 1 px
        (h,w,c) = rgb.shape
        centerX = []
        centerY = []
        width = []
        height = []

        for j in range(3):
            centerX.append((gt_bbox[j,0] + gt_bbox[j,2]) / 2)
            centerY.append((gt_bbox[j,1] + gt_bbox[j,3]) / 2)
            width.append(gt_bbox[j,2] - gt_bbox[j,0])
            height.append(gt_bbox[j,3] - gt_bbox[j,1])
        
            start_point = (int(gt_bbox[j,0]*w), int(gt_bbox[j,1]*h))
            end_point = (int(gt_bbox[j,2]*w), int(gt_bbox[j,3]*h))
            # cv2.rectangle(rgb, start_point, end_point, color, thickness) # add red bbox in image

        cv2.imshow('show_image', rgb)
        img_path = 'ab_10/BS_mix/images/{}_{:04d}.jpg'.format(plant, i)
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        # cv2.imwrite(img_path, rgb)
        cv2.waitKey(1)
        # time.sleep(0.1)

        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if not os.path.exists(os.path.dirname(label_path)):
            os.makedirs(os.path.dirname(label_path))
        with open(label_path, 'w') as f:
            for row in range(3):
                f.write("0 {} {} {} {}\n".format(centerX[row], centerY[row], width[row], height[row]))

        if cv2.waitKey(1) == 27:
            break

    # df = pd.DataFrame(bbox)
    # df.to_csv('bbox/{}_bbox.csv'.format(plant), index=False)



