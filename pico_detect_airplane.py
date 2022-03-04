#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import traceback

# from IPython.display import Image
import matplotlib.pyplot as plt

# import roslib
# GP_AIRPLANE_TEMP_PATH = roslib.packages.get_pkg_dir("hma_hsr_wrs_pkg") + "/io/airplane"


# 膨張処理
def expansion(src, ksize=4):
    h, w = src.shape
    dst = src.copy()

    for y in range(0, h):
        for x in range(0, w):
            # 近傍に白い画素が2つ以上あれば、注目画素を白色に塗り替える
            roi = src[y-ksize:y+ksize+1, x-ksize:x+ksize+1]
            if np.count_nonzero(roi) > 5:
                dst[y][x] = 255

    return dst

# 収縮処理
def contraction(src, ksize=11):
    h, w = src.shape
    dst = src.copy()

    for y in range(0, h):
        for x in range(0, w):
            # 近傍に黒い画素が1つでもあれば、注目画素を黒色に塗り替える
            roi = src[y-ksize:y+ksize+1, x-ksize:x+ksize+1]
            if roi.size - np.count_nonzero(roi) > 0:
                dst[y][x] = 0

    return dst

# トリミング処理
def trim(src, trim_size_x=15, trim_size_y=15):

    h, w = src.shape
    dst = src.copy()

    for y in range(0, h):
        for x in range(0, w):
            if y < trim_size_y or y > h - trim_size_y:
                dst[y][x] = 0
            if x < trim_size_x or x > w - trim_size_x:
                dst[y][x] = 0

    return dst

def estimate_grasppose_airplane(image_gray, image_edge):
    retval, image_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 輪郭の検出
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frontback_flag = "front"

    # 裏向きは膨張と収縮のパラメータを変更する
    if len(contours) <= 2:
        print("Retry edge")

        image_con = image_edge.copy()
        image_con = cv2.cvtColor(image_con, cv2.COLOR_BGR2GRAY)

        image_con = expansion(image_con, ksize = 5)
        image_con = contraction(image_con, ksize = 11)
        image_con = trim(image_con)

        image_gray = image_con.copy()
        retval, image_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # これでも正常に領域が取れない場合は、unknownと判定する

    if len(contours) <=2 or len(contours) >=9: # unknown判定
        print("Cannot recognize edge. It is not airplane.")
        return (999, 999), 999, frontback_flag

    # 最大の領域と二番目の領域を推定
    max_id, second_id = compare_area(contours)
    max_area = cv2.contourArea(contours[max_id])
    second_area = cv2.contourArea(contours[second_id])

    if second_area >= 800:
        frontback_flag = "back"
        print("This airplane is backward!!!!")
        tmp = max_id
        max_id = second_id
        second_id = tmp
        max_area = cv2.contourArea(contours[max_id])
        second_area = cv2.contourArea(contours[second_id])

    if max_id == 999: # areasizeがあまりに離れている場合はunknown判定する
        print("Area size default. It is unknown.")
        return (999, 999), 999, frontback_flag

    # 各領域の重心を求めて、その角度を算出
    pt1 = get_center(contours, max_id)
    pt2 = get_center(contours, second_id)

    print("tail_areasize", max_area, "head_areasize", second_area, "tail_position", pt1, "head_position", pt2)

    # yの差分がマイナスのときは処理を変える
    # 角度は右向きが0度で時計回り
    if (pt2[1] - pt1[1]) >= 0:
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    else:
        angle = math.atan2(pt1[1] - pt2[1], pt2[0] - pt1[0])
        angle = -angle
    angle_deg = angle * 60


    pt = get_point(image_gray, angle_deg, frontback_flag, contours[max_id])

    return pt, angle_deg, frontback_flag

def get_center(contours, id):
    maxCont = contours[id]
    mu = cv2.moments(maxCont)
    x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])

    return x, y

def get_point(image, angle_deg, frontback_flag, contours):
    angle_rad = (angle_deg / 180.0) * np.pi

    height = image.shape[0]
    width = image.shape[1]

    #回転のパラメータ
    center = (int(width/2), int(height/2))
    scale = 1.0

    # 変換行列を作成
    trans = cv2.getRotationMatrix2D(center, angle_deg , scale)

    #アフィン変換
    new_image_gray = cv2.warpAffine(image, trans, (width,height))

    # 輪郭の検出
    retval, image_bw = cv2.threshold(new_image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_id, second_id = compare_area(contours)

    # 後ろ向きの際はidを入れ替える
    if frontback_flag == "back":
        max_id = second_id

    # エッジ領域から右端を抽出
    pt = get_rightedge(new_image_gray, contours[max_id], frontback_flag)

    # アフィン変換前の画素と対応付ける
    pt_trans = Correspondence(image, pt, trans)

#     # debug用
#     cv2.circle(new_image_gray, (pt[0], pt[1]), 3, (255, 0, 0), thickness=-1)
#     print("回転後")
#     plt.imshow(new_image_gray)
#     plt.show()

    return pt_trans

def get_rightedge(image, contours, frontback_flag):

    image_gray = image.copy()

    if frontback_flag == "front":
        max_x = 0
        pt = (0, 0)
        for i in range(len(contours)):
            x = contours[i][0][0]
            y = contours[i][0][1]
            if max_x < x:
                max_x = x
                pt = (x - 8, y - 10)

    elif frontback_flag == "back":
        max_x = 0
        pt = (0, 0)
        for i in range(len(contours)):
            x = contours[i][0][0]
            y = contours[i][0][1]
            if max_x < x:
                max_x = x
                pt = (x - 8, y - 10)

    return pt

def compare_area(contours):
    # 輪郭の面積比較
    max_area = 0
    second_area = 0
    max_id = 0
    second_id = 0

    # 面積が最大の領域と2番めの領域を算出
    for num in range(len(contours)):
        if cv2.contourArea(contours[num]) < 3000: # 全体を領域と認識してしまうため、閾値以上の面積を持つ領域は無視する
            if max_area < cv2.contourArea(contours[num]):
                second_area = max_area
                second_id = max_id
                max_area = cv2.contourArea(contours[num])
                max_id = num

            elif second_area < cv2.contourArea(contours[num]):
                second_area = cv2.contourArea(contours[num])
                second_id = num

            if second_area >= max_area:
                second_area = 0
                max_area = 0

        # else: # 全体の領域があまりにも違う場合はunknown判定にする
        #     all_area = cv2.contourArea(contours[num])
        #     print("all_area", all_area)
        #     if 35000 > all_area > 40000:
        #         print("area size default. It is unknown.")
        #         return 999, 999

    return max_id, second_id

def Correspondence(image, pt, trans):

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            # 各画素をアフィン変換し、変換後の座標を得る
            tmp_pt = (w, h, 1)
            check = np.dot(trans, tmp_pt)
            check = (int(check[0]), int(check[1]))

            # 変換後の座標がパターン認識の座標と等しい（近い）ければ、そこが対応付けられた変換前の座標
            if pt[0]-2 < check[0] <= pt[0] and pt[1] <= check[1] < pt[1]+2:
                corr_pt = (tmp_pt[0], tmp_pt[1])
                if check == pt: # 同一の座標を見つけたらそこで終了
                    return corr_pt

    return corr_pt

if __name__ == '__main__':

    for i in range(1, 4):
        read_path = '/home/yuga/yuga_ws/wrs_ws/airplane_true/edge_0' + str(i) + '.jpg'

        # image_edgeがエッジ検出したあとの画像
        image_edge = cv2.imread(read_path)
        image_gray = cv2.cvtColor(image_edge, cv2.COLOR_BGR2GRAY)

        # 画像を膨張・収縮・トリミングする
        image_gray = expansion(image_gray, ksize=4)
        cv2.imshow("after expan", image_gray)
        image_gray = contraction(image_gray, ksize=11)
        cv2.imshow("after contraction", image_gray)
        image_gray = trim(image_gray)
        cv2.imshow("after trim", image_gray)

        # 膨張等の処理を施したグレー画像と、エッジ検出をしたカラー画像を入力とする
        grasp_pos, angle, frontback_flag = estimate_grasppose_airplane(image_gray, image_edge)

        # 例外処理が置きた場合はunknownと判定する
        if angle == 999:
            print("It is unknown object.")

        # 確認と表示用
        image_result = cv2.circle(image_edge, (grasp_pos[0], grasp_pos[1]), 3, (255, 0, 0), thickness=-1)
        print("grasp_position", grasp_pos, "angle", angle)
        # plt.imshow(image_result)
        # plt.show()
        cv2.imshow("image_result", image_result)
        cv2.waitKey(0)


# 変更箇所
# expansion ksize = 4
# expansion roi > 5

# contraction ksize = 11
