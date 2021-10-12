#!/usr/bin/env python
# -*- coding: utf-8 -*-


#==================================================

## @file check_direction.py
## @author Yuga YANO
## @brief OrientationItemの方向推定用クラス

#==================================================


import cv2
# from PIL import Image, ImageFilter
import PIL.Image
import numpy as np
import traceback

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

class CheckDirection:
    def __init__(self):
        return


    #==================================================

    ## @fn cv2pil
    ## @brief opencvのimageをpillowに変換
    ## @param image cv2型のカラー画像
    ## @return new_image pillow型のカラー画像

    #==================================================+
    def cv2pil(self, image):
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = PIL.Image.fromarray(new_image)

        return new_image


    #==================================================

    ## @fn get_objectedge
    ## @brief 物体の左端と右端を算出
    ## @param img pillow型のカラー画像
    ## @return left_edge 物体の左端(x, y)
    ## @return right_edge 物体の右端(x, y)

    #==================================================
    def get_objectedge(self, img):
        left_edge, right_edge = {'x':img.width, 'y': 0}, {'x':0, 'y': 0}
        # print(type(left_edge['x']))
        r, g, b = img.convert("RGB").split()

        # マスク部分ではない画素情報を求めることで、物体の左端と右端を算出
        for x in range(img.width):
            for y in range(0, img.height):
                if r.getpixel((x, y)) != 0 and g.getpixel((x, y)) != 0 and b.getpixel((x, y)) != 0:
                    if left_edge['x'] >= x:
                        left_edge['x'], left_edge['y'] = x, y
                    if right_edge['x'] <= x:
                        right_edge['x'], right_edge['y'] = x, y

        # print('edge of object, left = {}, right = {}' .format(left_edge, right_edge))
        return left_edge, right_edge



    #==================================================

    ## @fn detectCutleryOrientation
    ## @brief Cutleryオブジェクトの向き推定
    ## @param img pillow型のカラー画像
    ## @return flag 柄の部分がどちらにあるかを(left, right)で返答
    ##               柄が右側ならright, 左側ならleftで返答

    #==================================================
    def detectCutleryOrientation(self, img):
        r, g, b = img.convert("RGB").split()
        # 物体の左端と右端を取得
        left_edge, right_edge = self.get_objectedge(img)

        # 物体の中間地点を算出 ((右端 - 左端) / 2) + 左端)
        width_th = ((right_edge['x'] - left_edge['x']) / 2.0) + left_edge['x']

        # 左と右の赤色と認識した画素の量を算出
        right, left = 0, 0
        for x in range(left_edge['x'], right_edge['x']):
            for y in range(0, img.height):
                if r.getpixel((x, y)) > 100 and g.getpixel((x, y)) < 40 and b.getpixel((x, y)) < 40:
                    if width_th > x:
                        left += 1
                    else:
                        right += 1

        # どちらに赤色が多かったかの判定
        if right > left:
            flag = "right"
        else:
            flag = "left"

        # print('handle postion = {}' .format(flag))
        rospy.loginfo("handle　position = " + flag)
        return flag



    #==================================================

    ## @fn detectMarkerOrientation
    ## @brief Cutleryオブジェクトの向き推定
    ## @param img pillow型のカラー画像
    ## @return flag 柄の部分がどちらにあるかを(left, right)で返答
    ##             キャップでない側が右側ならright, 左側ならleftで返答

    #==================================================
    def detectMarkerOrientation(self, img):
        r, g, b = img.convert("RGB").split()
        # 物体の左端と右端を取得
        left_edge, right_edge = self.get_objectedge(img)

        left, right = 0, 0
        # 左端と右端の周辺画素におけるRGB値の総和を算出
        # 左端周辺の画素値計算
        for x in range(left_edge['x'], left_edge['x'] + 10):
            for y in range(left_edge['y'] - 10, left_edge['y'] + 10):
                left += (r.getpixel((x, y)) + g.getpixel((x, y)) + b.getpixel((x, y)))

        # 右端周辺の画素値計算
        for x in range(right_edge['x'] - 10, right_edge['x']):
            for y in range(right_edge['y'] - 10, right_edge['y'] + 10):
                right += (r.getpixel((x, y)) + g.getpixel((x, y)) + b.getpixel((x, y)))

        if left > right:
            flag = 'left'
        else:
            flag = 'right'

        # print('nocap postion = {}' .format(flag))
        rospy.loginfo("nocap postion = " + flag)
        return flag




    def detect_orientation_floor(self, img, obj_id=26):
        method = eval('cv2.TM_CCOEFF')
        # print(img.shape())
        # forkの場合
        if obj_id == 26:
            # right template Matching
    #         img = cv2.imread('pattern/fork_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/fork_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/fork_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        # spoonの場合
        elif obj_id == 27:
            # right template Matching
    #         img = cv2.imread('pattern/spoon_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/spoon_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/spoon_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        # markerの場合
        else:
            # right template Matching
    #         img = cv2.imread('pattern/spoon_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/marker_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/marker_yuka.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        if left_match > right_match:
            orientation = "left"
        else:
            orientation = "right"
        # print(orientation)

        return orientation


    def detect_orientation_table(self, img, obj_id=26):
        method = eval('cv2.TM_CCOEFF')
        # forkの場合
        if obj_id == 26:
            # right template Matching
    #         img = cv2.imread('pattern/fork_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/fork_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/fork_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        # spoonの場合
        elif obj_id == 27:
            # right template Matching
    #         img = cv2.imread('pattern/spoon_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/spoon_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/spoon_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        # markerの場合
        else:
            # right template Matching
    #         img = cv2.imread('pattern/spoon_yuka_left.jpg')
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/right/marker_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            right_match = max_val

            # left template Matching
            template_image_path = "/home/hma/ros_ws/hma/hma_ws/src/robot_pkgs/tasks/hma_hsr_wrs_pkg/script/src/task1/pattern/left/marker_tukue.jpg"
            template = cv2.imread(template_image_path)
            res = cv2.matchTemplate(img, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            left_match = max_val

        if left_match > right_match:
            orientation = "left"
        else:
            orientation = "right"
        # print(orientation)

        return orientation


    def detect_orientation_4patternmatch(self, img, place, obj_id):
        # y = 170
        # h = 350
        # x = 150
        # w = 490

        # img_trim = img[y:y+h, x:x+w]

        if place == "floor":
            y = 170
            h = 250
            x = 150
            w = 300

            img_trim = img[y:y+h, x:x+w]
            orientation = self.detect_orientation_floor(img_trim)
        else:
            y = 170
            h = 200
            x = 190
            w = 280

            img_trim = img[y:y+h, x:x+w]
            orientation = self.detect_orientation_table(img_trim)
        
        return orientation

# if __name__ == '__main__':
#     img = cv2.imread('pattern/bk_pattern/yuka/fork.jpg')
#     # place = "table" or "floor"
#     detect_orientation_4patternmatch(img, place="table", obj_id=26)


# テスト用
# if __name__ == '__main__':
    # img_path = '/home/rc21/robcup2021_ws/images/marker_0.jpg'
    # img = cv2.imread(img_path)

    # 始めからpillow形式で読み込んだ場合は必要なし
    # img_pil = cv2pil(img)

    # try:
    #     detectMarkerOrientation(img_pil)
    #     detectCutleryOrientation(img_pil)
    # except:
    #     traceback.print_exc()
