#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import

import cv2
import numpy as np
import math
import traceback

import matplotlib.pyplot as plt

# %% compare_area
# 領域すべてを比較し，1番大きい領域と2番目に大きい領域のidを返す

def compare_area(contours):
    # 輪郭の面積比較
    max_area = 0
    second_area = 0
    max_id = 0
    second_id = 0

    # 面積が最大の領域と2番めの領域を算出
    for num in range(len(contours)):
        if cv2.contourArea(contours[num]) < 30000: # 全体を領域と認識してしまうため、閾値以上の面積を持つ領域は無視する
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

# %% get point
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

def get_center(contours, id):
    maxCont = contours[id]
    mu = cv2.moments(maxCont)
    x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])

    return x, y

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

def Correspondence(image, pt, trans):

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            # 各画素をアフィン変換し、変換後の座標を得る
            tmp_pt = (w, h, 1)
            check = np.dot(trans, tmp_pt)
            check = (int(check[0]), int(check[1]))

            # 変換後の座標がパターン認識の座標と等しい（近い）ければ、そこが対応付けられた変換前の座標
            if pt[0]-20 < check[0] <= pt[0]+20 and pt[1]-20 <= check[1] < pt[1]+20:
                corr_pt = (tmp_pt[0], tmp_pt[1])
                if check == pt: # 同一の座標を見つけたらそこで終了
                    return corr_pt

    return corr_pt


# %% detect airplane

def estimate_grasppose_airplane(image_bgr, threshold_r=40):
    # write_path = "C:/Users/fryuz/prog/hma/imgs/airplanes/temp"
    # 色領域の抜き出し（R平面を用いる）
    image_b, _, image_r = cv2.split(image_bgr)
    _, r_binary = cv2.threshold(image_r, threshold_r, 255, cv2.THRESH_BINARY)
    r_binary = cv2.bitwise_not(r_binary)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(r_binary, kernel, iterations=1)
    image_mask = cv2.dilate(erosion, kernel, iterations=6)

    # 輪郭の検出
    contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frontback_flag = "front"

    # # 裏向きは膨張と収縮のパラメータを変更する
    # if len(contours) <= 2:
    #     print("Retry edge")
    #
    #     image_con = image_edge.copy()
    #     image_con = cv2.cvtColor(image_con, cv2.COLOR_BGR2GRAY)
    #
    #     image_con = expansion(image_con, ksize = 5)
    #     image_con = contraction(image_con, ksize = 11)
    #     image_con = trim(image_con)
    #
    #     image_gray = image_con.copy()
    #     retval, image_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     # これでも正常に領域が取れない場合は、unknownと判定する

    if len(contours) <2 or len(contours) >=9: # unknown判定
        print("Cannot recognize edge. It is not airplane.")
        return (999, 999), 999, frontback_flag

    # 最大の領域と二番目の領域を推定
    max_id, second_id = compare_area(contours)
    # print(max_id, second_id)
    max_area = cv2.contourArea(contours[max_id])
    second_area = cv2.contourArea(contours[second_id])

    # 各領域の重心を求めて、その角度を算出
    pt1 = get_center(contours, max_id)
    pt2 = get_center(contours, second_id)

    # yの差分がマイナスのときは処理を変える
    # 角度は右向きが0度で時計回り
    if (pt2[1] - pt1[1]) >= 0:
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    else:
        angle = math.atan2(pt1[1] - pt2[1], pt2[0] - pt1[0])
        angle = -angle
    angle_deg = angle * 60

    pt = get_point(image_r, angle_deg, frontback_flag, contours[max_id])
    # temp = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(temp, contours, -1, (255, 255, 0), 3)
    # cv2.imwrite(write_path + "_mask_visualize.jpg", temp)

    print("tail_areasize", max_area, "head_areasize", second_area, "tail_position", pt1, "head_position", pt2)
    print("angle", angle_deg, "point", pt)

    return pt, angle_deg, frontback_flag


# %% do
if __name__ == "__main__":
    # for i in range(19, 20):
    read_path = "C:/Users/fryuz/prog/hma/imgs/airplanes/frame0019.jpg"
    img_rgb = cv2.imread(read_path, 1)
    pt, angle, flag = estimate_grasppose_airplane(img_rgb)
