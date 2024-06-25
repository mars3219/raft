import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_flo_file(filename):
    with open(filename, 'rb') as f:
        # Read the magic number
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError('Invalid .flo file format')
        
        # Read the width and height of the flow data
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        
        # Read the flow data
        flow_data = np.fromfile(f, np.float32, count=2*width*height)
        flow_data = flow_data.reshape((height, width, 2))
        
    return flow_data


def visualize_flow(flow_data):
    h, w = flow_data.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculate magnitude and angle of flow
    mag, ang = cv2.cartToPolar(flow_data[..., 0], flow_data[..., 1])
    
    # Set hue based on flow direction
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Set value based on flow magnitude
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to RGB for display
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rgb_flow)
    # plt.title('Optical Flow Visualization')
    # plt.show()

    cv2.imshow('Optical Flow', rgb_flow)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # .flo 파일 읽기
    flow_data = read_flo_file('/workspace/RAFT/00001_flow.flo')

    # 옵티컬 플로우 시각화
    visualize_flow(flow_data)
