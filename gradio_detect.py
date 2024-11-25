import gradio as gr
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

def detect_large_regions(image, w_min, w_max, h_min, h_max, threshold_factor=1.5):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值分割将图像转换为二值图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 进行连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 计算平均面积
    avg_area = np.mean(stats[1:, cv2.CC_STAT_AREA])  # 排除背景
    area_threshold = avg_area * threshold_factor  # 定义大面积阈值

    area_new = []

    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_threshold:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # 更严格限制检测断裂的判断，确保是显著的断裂部分
            if w_min < w < w_max and h_min < h < h_max:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'Defect', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                area_new.append(area)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    avg_area_new = np.mean(area_new) if area_new else 0
    return image, f"平均大面积: {avg_area_new}"

def main(image, w_min, w_max, h_min, h_max):
    if image is None:
        raise gr.Error("请先上传图片")
    result, avg_area = detect_large_regions(image, w_min, w_max, h_min, h_max)
    return result, avg_area

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>尺寸控制的缺陷检测</h1>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图像")
            w_min_slider = gr.Slider(0, 500, value=100, step=1, label="最小宽度 (w_min)")
            w_max_slider = gr.Slider(0, 500, value=200, step=1, label="最大宽度 (w_max)")
            h_min_slider = gr.Slider(0, 500, value=100, step=1, label="最小高度 (h_min)")
            h_max_slider = gr.Slider(0, 500, value=200, step=1, label="最大高度 (h_max)")
            btn = gr.Button("提交")
        with gr.Column():
            image_output = gr.Image(type="pil", label="输出图像")
            avg_area_output = gr.Textbox(label="平均面积")

    # 将按钮与函数连接
    btn.click(fn=main, inputs=[image_input, w_min_slider, w_max_slider, h_min_slider, h_max_slider], outputs=[image_output, avg_area_output])

demo.launch()
