import numpy as np
from matplotlib import pyplot as plt
from common import *
# feel free to include libraries needed


def homography_transform(X, H):
    # TODO
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)

    X = np.hstack((X, np.ones((len(X), 1))))
    Y_temp = X @ H.T
    Y = Y_temp / Y_temp[:, 2:3]
    return Y


def fit_homography(XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    N = len(XY)
    X = XY[:, :2]
    Y = XY[:, 2:]
    X = np.hstack((X, np.ones((len(X), 1))))
    Y = np.hstack((Y, np.ones((len(Y), 1))))
    A = []
    for i in range(N):
        l1 = np.concatenate(([0, 0, 0], -1 * X[i], Y[i][1] * X[i]))
        l2 = np.concatenate((X[i], [0, 0, 0], -1 * Y[i][0] * X[i]))
        A.append(l1)
        A.append(l2)
    A = np.vstack(A)
    eigenValues, eigenVectors = np.linalg.eig(A.T @ A)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    v = eigenVectors[:, -1]
    H = np.vstack((v[:3].T, v[3:6].T, v[6:].T))
    return H


def p1():
    # code for Q1.2.3 - Q1.2.5
    # 1. load points X from p1/transform.npy
    data = np.load(p1_path+'p1/transform.npy')
    X = data[:,:2]
    old_X = X
    Y = data[:,2:]
    X = np.hstack((X, np.ones((len(X),1)))) 
    
    # print(rst[0].T)
    # 2. fit a transformation y=Sx+t
    rst = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond = None)
    # 3. transform the points
    Y_temp = X @rst[0]
    print("S", rst[0][0:2], "\nt", rst[0][-1])
    # 4. plot the original points and transformed points
    plt.scatter(old_X[:, 0], old_X[:, 1], label = "x",  c="red")  # X
    plt.scatter(Y[:, 0], Y[:, 1], label = "y", c="green")  # Y
    plt.scatter(Y_temp[:, 0], Y_temp[:, 1], label=r"$\^y$", c="blue")  # Y_temp
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(path+'125.jpg')
    plt.close()
    # print(transformed)
    # code for Q1.2.6 - Q1.2.8
    case = 8  # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load(p1_path+'p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography()
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print("For case", i,  H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transsform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:, :2], H)
        # 4. Visualize points as three images in one figure
        # the following code plot figure for you

        fig = plt.figure(figsize=(20,8))

        plt.subplot(1,3,1)
        plt.scatter(XY[:, 1], XY[:, 0], c="black")
        plt.title("Original Points")
        plt.grid()

        plt.subplot(1,3,2)
        plt.scatter(XY[:, 3], XY[:, 2], c="green")  
        plt.title("Target Points")
        plt.grid()

        plt.subplot(1,3,3)
        plt.scatter(Y_H[:, 1], Y_H[:, 0], c="red")  
        plt.title("Points after homography transformation")
        plt.grid()
        plt.savefig(path+'./case_'+str(i))
        # plt.show()

def q_1_2(imgleft, imgright, savename): 

    save_img(imgleft, '{}q1_{}_left_grayscale.jpg'.format(path, savename))
    save_img(imgright, '{}q1__{}_right_grayscale.jpg'.format(path, savename))

    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    keypoints_sift_left, descriptors = sift.detectAndCompute(imgleft, None)
    keypoints_sift_right, descriptors = sift.detectAndCompute(imgright, None)

    imgl = cv2.drawKeypoints(imgleft, keypoints_sift_left, None)
    imgr = cv2.drawKeypoints(imgright, keypoints_sift_right, None)
    save_img(imgl, '{}q2{}_left_sift.jpg'.format(path, savename))
    save_img(imgr, '{}q2{}_right_sift.jpg'.format(path, savename))


    keypoints_surf_left, descriptors = surf.detectAndCompute(imgleft, None)
    keypoints_surf_right, descriptors = surf.detectAndCompute(imgright, None)

    imgl = cv2.drawKeypoints(imgleft, keypoints_surf_left, None)
    imgr = cv2.drawKeypoints(imgright, keypoints_surf_right, None)
    save_img(imgl, '{}q2{}_left_surf.jpg'.format(path, savename))
    save_img(imgr, '{}q2{}_right_surf.jpg'.format(path, savename))

#@title
def stitchimage(imgleft, imgright):
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv
  
    sift = cv2.xfeatures2d.SIFT_create()
    key_point_left,descriptors_left = sift.detectAndCompute(imgleft,None)
 
    # imgleft=cv2.drawKeypoints(imgleft,key_point_left,imgleft)
    
    sift = cv2.xfeatures2d.SIFT_create()
    key_point_right,descriptors_right = sift.detectAndCompute(imgright,None)

    key_points_temp_left = np.array([[p.pt[0],p.pt[1]] for p in key_point_left])
    tl = key_points_temp_left
    X = key_points_temp_left
    key_points_temp_left = descriptors_left[:,None,:]
    key_points_temp_right = np.array([[p.pt[0],p.pt[1]] for p in key_point_right])
    tr = key_points_temp_right
    Y = key_points_temp_right
    key_points_temp_right = np.transpose(descriptors_right[:,None,:],(1,0,2))
    descriptors_left = ((descriptors_left - np.mean(descriptors_left, axis = 0, keepdims=True))/np.std(descriptors_left,axis = 0, keepdims=True))
    descriptors_right = ((descriptors_right - np.mean(descriptors_right, axis = 0, keepdims=True))/np.std(descriptors_right,axis = 0, keepdims=True))
    norm_left = np.sum(descriptors_left**2, axis = 1, keepdims = True)
    norm_right = np.sum(descriptors_right**2, axis = 1, keepdims = True)
    dist = (norm_left + norm_right.T -2*(descriptors_left @ descriptors_right.T))**0.5

    # 2. select paired descriptors

    paired_1 = np.amin(dist,axis = 1)
    print(np.argpartition(dist,2))
    paired_2args = np.argpartition(dist,2)[:,2]
    paired_2= dist[np.arange(len(dist)),paired_2args]
    print(paired_1.shape, paired_2.shape)
    s = paired_1/paired_2 < 0.5

    print(paired_1, paired_2)
  

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers
    best_homography, best_fit_numbers = None, -1
    bestd = 0
    id_x = np.nonzero(s)[0]

    id_y = np.argmin(dist, axis = 1)[id_x]
    assert(len(id_x) == len(id_y))
    data = np.hstack((X[id_x,:],Y[id_y,:]))
    minimal_dist = None
    for i in range(400):
        id_s = np.random.randint(0,len(id_x),size = 50)
        H = fit_homography(data)
        Y_h = homography_transform(X[id_x[id_s],:],H)[:,:2]
        # print(n.shape,data.shape,d.shape)
        dist = np.linalg.norm(Y_h - Y[id_y[id_s],:],axis = 1)
        ct = np.sum(dist < 20)
        if ct > best_fit_numbers:
            best_homography = H
            best_fit_numbers = ct
            minimal_dist = dist
   
    print("best fit inliers",best_fit_numbers)
    print("best_homography", best_homography)
    print("residual",np.mean(minimal_dist[minimal_dist < 20]))
    match_idx = (minimal_dist < 20)
    key_point1 = []
    key_point2 = []
    nidx  =  np.nonzero(match_idx)[0]

    match1to2 = []
    midx = 0
    for i in nidx:
        key_point1.append(cv2.KeyPoint(tl[id_x[i]][0],tl[id_x[i]][1],1))
        key_point2.append(cv2.KeyPoint(tr[id_y[i]][0],tr[id_y[i]][1],1))
        match1to2.append(cv2.DMatch(midx,midx,dist[i]))
        midx += 1
    
    match = None
    match = cv2.drawMatches(imgleft,key_point1,imgright,key_point2,match1to2 ,match )
    

    # 4. warp one image by your transformation
    #    matrix
    #
    #    Hint:
    #    a. you can use function of opencv to warp image
    #    b. Be careful about final image size
    best_homography = best_homography / best_homography[2,2]
    translate = np.array([[ 1 , 0 , 0],[ 0 , 1 , 0],[ 0 , 0 ,  1 ]])
    H_inv = np.linalg.inv(best_homography)
    print(H_inv)
    warped = cv2.warpPerspective(imgright,H_inv, (600,900))
    save_img(warped, '/.warped.jpg')
    
    # 5. combine two images, use average of them
    #    in the overlap area
 
    warpleftc = cv2.warpPerspective(imgleft,translate @ best_homography, (2500,1500))
    warprightc = cv2.warpPerspective(imgright,translate.astype(np.float) , (2500,1500))
    img = warpleftc.astype(np.int32) + warprightc.astype(np.int32)
    print("sum",np.sum(warpleftc, axis = 2).shape)
    overlap = np.logical_and(np.sum(warpleftc, axis = 2),np.sum(warprightc, axis = 2))
    print(overlap.shape,img[overlap,:].shape)
    img[overlap,:]  -= (0.5*img[overlap,:]).astype(np.int32)
    cv2_imshow(img)
    return img, best_homography



def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_img(p1)
    imgright = read_img(p2)
    iml = cv2.imread(p1, cv2.IMREAD_COLOR)
    imr = cv2.imread(p2, cv2.IMREAD_COLOR)
    output, H = stitchimage(imgleft, imgright, iml, imr, savename)
    return H




if __name__ == "__main__":
    # Problem 1
    # p1()

    # Problem 2
    p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

