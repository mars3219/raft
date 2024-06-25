import sys
sys.path.append('core')

import time
import argparse
import torch
import numpy as np
import cv2
from raft import RAFT
from utils.utils import InputPadder
import torchvision.transforms as transforms

RW = 640
RH = 320

# Optical flow 계산 함수
def compute_flow(model, image1, image2):
    transform = transforms.ToTensor()

    image1 = transform(image1).unsqueeze(0).to('cuda')
    image2 = transform(image2).unsqueeze(0).to('cuda')
    
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    flow_up = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
    return flow_up

# 화살표로 optical flow 시각화 함수
def draw_flow_arrows(image, flow, step=32):
    h, w = image.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+2*fx, y+2*fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    vis = image.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
    return vis

def main(args):
    # RAFT 모델 로드
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to('cuda')
    model.eval()

    # 동영상 파일 열기
    cap = cv2.VideoCapture('/workspace/RAFT/fig.mp4')
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (RW, RH))

    while(cap.isOpened()):
        ret, frame2 = cap.read()
        if not ret:
            break
        
        frame2 = cv2.resize(frame2, (RW, RH))
        st = time.time()
        # Optical flow 계산
        flow = compute_flow(model, frame1, frame2)
        ed = time.time()
        
        # Optical flow 화살표로 시각화
        flow_arrows = draw_flow_arrows(frame2, flow)
        
        print(f"{1 / (ed-st): .2f} fps")
        # 결과 출력
        cv2.imshow('Optical Flow', flow_arrows)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
        
        frame1 = frame2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/workspace/RAFT/models/raft-small.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_false', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    main(args)
