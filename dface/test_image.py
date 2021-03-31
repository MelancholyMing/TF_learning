import cv2
from dface.core.detect import create_mtcnn_net, MtcnnDetector
from dface.core import vision


if __name__ == '__main__':

    # refer to your local model path 
    p_model = "./model_store/pnet_epoch.pt"
    r_model = "./model_store/rnet_epoch.pt"
    o_model = "./model_store/onet_epoch.pt"

    #use cpu version set use_cuda=False, if you want to use gpu version set use_cuda=True
    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model, r_model_path=r_model, o_model_path=o_model, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./test.jpg")
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align

    vision.vis_face(img2,bboxs,landmarks)
