import cv2
import numpy as np

mouseDrawing = False
startX, startY, endX, endY = -1, -1, -1, -1
minMatchCount = 1
viewName = 'input image'

# 특징 추출기 생성
brisk = cv2.BRISK_create()
orb = cv2.ORB_create()

# BruteForce 기본 매칭 함수
def bruteForce(img1, img2):
    methods = [
        (brisk, cv2.NORM_HAMMING, 'bf_brisk'),
        (orb, cv2.NORM_HAMMING, 'bf_orb')
    ]
    
    flag = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    for (method, norm, name) in methods:
        keyP1, des1 = method.detectAndCompute(img1, None)
        keyP2, des2 = method.detectAndCompute(img2, None)
        
        bf = cv2.BFMatcher_create(norm)
        matches = bf.match(des1, des2)

        res = cv2.drawMatches(img1, keyP1, img2, keyP2, matches, None, flags=flag)
        cv2.imshow(name, res)

# FLANN 기반 매칭 함수
def flann(img1, img2):
    flannIndexKdTree = 1
    indexParamsKdTree = dict(algorithm=flannIndexKdTree, trees=5)
    searchParams = dict(checks=50)
    flannKdTree = cv2.FlannBasedMatcher(indexParamsKdTree, searchParams)

    methods = [
        (brisk, 'flann_brisk'),
        (orb, 'flann_orb')
    ]

    for (method, name) in methods:
        keyP1, des1 = method.detectAndCompute(img1, None)
        keyP2, des2 = method.detectAndCompute(img2, None)

        matches = flannKdTree.knnMatch(des1, des2, k=2)
        good = []
        
        # 좋은 매칭 점 선택
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        res = cv2.drawMatches(img1, keyP1, img2, keyP2, good, None)
        cv2.imshow(name, res)

# 이미지 로딩 및 매칭 실행
img1 = cv2.imread('OPENCV/Enb6u0aBS.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('OPENCV/r.png', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인하세요.")
else:
    bruteForce(img1, img2)
    # flann(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
