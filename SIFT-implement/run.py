import cv2 
import pickle
import matplotlib.pyplot as plt
import pysift
import time


# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints
def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

'''
# Define a list of images the way you like
imageList = ["taj1.jpg","taj2.jpg","eiffel1.jpg","eiffel2.jpg","liberty1.jpg","liberty2.jpg","robert1.jpg","tom1.jpg","ironman1.jpg","ironman2.jpg","ironman3.jpg","darkknight1.jpg","darkknight2.jpg","book1.jpg","book2.jpg"]

# We use grayscale images for generating keypoints
imagesBW = []
for imageName in imageList:
    imagePath = "data/images/" + str(imageName)
    imagesBW.append(imageResizeTrain(cv2.imread(imagePath,0))) # flag 0 means grayscale
'''

imageList = []
imagesBW = []
for i in range(5017):
    try:
        imgid = "{:04d}".format(i)
        dir_img0 = "/home/ddd/Downloads/Image-Local-Feature-Extraction-and-Matching/SIFT-implement/data/dataset_200/scene_" + str(i) + "/" + imgid + "_0.jpg"
        dir_img1 = "/home/ddd/Downloads/Image-Local-Feature-Extraction-and-Matching/SIFT-implement/data/dataset_200/scene_" + str(i) + "/" + imgid + "_1.jpg"
        imageList.append(dir_img0)
        imageList.append(dir_img1)
        imagesBW.append(imageResizeTrain(cv2.imread(dir_img0,0))) # flag 0 means grayscale
        imagesBW.append(imageResizeTrain(cv2.imread(dir_img1,0))) # flag 0 means grayscale
    except:
        imageList.pop()
        imageList.pop()

start = time.perf_counter()


############################
# Choice of implementation #
############################

'''
# main function: custom implementation
keypoints = []
descriptors = []
for i,image in enumerate(imagesBW):
    print("Starting for image: " + imageList[i])
    keypointTemp, descriptorTemp = pysift.computeKeypointsAndDescriptors(image)
    keypoints.append(keypointTemp)
    descriptors.append(descriptorTemp)
    print("  Number of keypoints is: ",len(keypointTemp))
    print("  Ending for image: " + imageList[i])
'''
# main function: opencv implementation
keypoints = []
descriptors = []
sift = cv2.SIFT_create()
for i,image in enumerate(imagesBW):
    print("Starting for image: " + imageList[i])
    keypointTemp, descriptorTemp = sift.detectAndCompute(image, None)
    keypoints.append(keypointTemp)
    descriptors.append(descriptorTemp)
    print("  Number of keypoints is: ",len(keypointTemp))
    print("  Ending for image: " + imageList[i])


end = time.perf_counter()

time_use = end - start

print("time use for extraction is: %f seconds"%(time_use))


# store keypoints an descripters
for i,keypoint in enumerate(keypoints):
    deserializedKeypoints = []
    #filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
    filepath = str(imageList[i].split('.')[0]) + ".kp.txt"
    for point in keypoint:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        deserializedKeypoints.append(temp)
    with open(filepath, 'wb') as fp:
        pickle.dump(deserializedKeypoints, fp)  

for i,descriptor in enumerate(descriptors):
    #filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
    filepath = str(imageList[i].split('.')[0]) + ".dp.txt"
    with open(filepath, 'wb') as fp:
        pickle.dump(descriptor, fp)


# Fentch keypoints and descripters for future use
def fetchKeypointFromFile(i):
    #filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
    filepath = str(imageList[i].split('.')[0]) + ".kp.txt"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint

def fetchDescriptorFromFile(i):
    #filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
    filepath = str(imageList[i].split('.')[0]) + ".dp.txt"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

start = time.perf_counter()

# knn matching
bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

end = time.perf_counter()

# calculate results for image pairs
def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [-1,-1,-1],
        flags=2
    )
    return matchPlot

def getPlotFor(i,j,keypoint1,keypoint2,matches):
    #image1 = imageResizeTest(cv2.imread("data/images/" + imageList[i]))
    #image2 = imageResizeTest(cv2.imread("data/images/" + imageList[j]))
    image1 = imageResizeTest(cv2.imread(imageList[i]))
    image2 = imageResizeTest(cv2.imread(imageList[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)


def calculateResultsFor(i,j):
    print("Comparision between files:")
    print("    ",imageList[i])
    print("    ",imageList[j])
    keypoint1 = fetchKeypointFromFile(i)
    descriptor1 = fetchDescriptorFromFile(i)
    keypoint2 = fetchKeypointFromFile(j)
    descriptor2 = fetchDescriptorFromFile(j)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(i,j,keypoint1,keypoint2,matches)
    print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    print(score)
    plt.imshow(plot)
    plt.axis('off')
    plt.show()

#calculateResultsFor(11,10)
    
def calculateResultsForScene(p):
    i = 2*p
    j = 2*p+1
    print("Comparision between files:")
    print("    ",imageList[i])
    print("    ",imageList[j])
    keypoint1 = fetchKeypointFromFile(i)
    descriptor1 = fetchDescriptorFromFile(i)
    keypoint2 = fetchKeypointFromFile(j)
    descriptor2 = fetchDescriptorFromFile(j)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(i,j,keypoint1,keypoint2,matches)
    #print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    print("Number of keypoints:")
    print("    First image: ",len(keypoint1))
    print("    Second image: ",len(keypoint2))
    print("Number of matches: ",len(matches))
    print("Similarity score of pictures is: ",score)
    plt.imshow(plot)
    plt.axis('off')
    plt.show()

calculateResultsForScene(11)

time_use = end - start

print("time use for matching is: %f seconds"%(time_use))