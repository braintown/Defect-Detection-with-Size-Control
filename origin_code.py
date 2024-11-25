import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值分割将图像转换为二值图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 进行连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    return image, labels, stats

def detect_large_regions(image_path, output_path, threshold_factor=1.5):
    image, labels, stats = preprocess_image(image_path)

    # 计算平均面积
    avg_area = np.mean(stats[1:, cv2.CC_STAT_AREA])  # 排除背景
    area_threshold = avg_area * threshold_factor  # 定义大面积阈值
    print("Adjusted area threshold:", area_threshold)
    print("Average area:", avg_area)

    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_threshold:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area_new = []
            area_new.append(area)


            # 更严格限制检测断裂的判断，确保是显著的断裂部分
            if 200 > w > 100 and h > 100:
                # print(f"w{w}")
                # print(f"h{h}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Defect', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    avg_area_new = np.mean(area_new)
    print(f"area_average{avg_area_new}")
    cv2.imwrite(output_path, image)

def main():
    image_path = '3.jpg'  # 输入图像路径
    output_path = 'output.jpg'  # 输出结果图像路径
    detect_large_regions(image_path, output_path)

if __name__ == "__main__":
    main()
