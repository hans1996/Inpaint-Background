import streamlit as st
import os
import numpy as np
import cv2
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()

# Vision Agent 套件
from vision_agent.tools import (
    load_image,
    florence2_sam2_instance_segmentation,
    flux_image_inpainting,              
    save_image
)

def process_mask(mask: np.ndarray, dilate_kernel_size: int = 5) -> np.ndarray:
    """处理遮罩以改善边缘效果"""
    # 转换为uint8格式
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 对遮罩进行膨胀操作，改善边缘覆盖
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # 转换回二值遮罩 (0 或 1)
    binary_mask = (dilated_mask > 127).astype(np.float32)
    
    return binary_mask

def enhance_prompt(base_prompt: str) -> str:
    """增强背景提示词"""
    # 添加更多细节和控制参数
    enhanced_prompt = f"{base_prompt}, professional product photography, studio lighting, 8k uhd, highly detailed, sharp focus, balanced composition"
    return enhanced_prompt

def segment_and_replace_background(
    image_path: str,
    background_prompt: str,
    output_path: str = "bracelet_with_background.png",
) -> np.ndarray:
    """
    改进的背景替换函数
    """
    # 1. 载入图片
    image = load_image(image_path)

    # 2. 检测手链
    segmentation_result = florence2_sam2_instance_segmentation("bracelet", image)
    if not segmentation_result:
        raise ValueError("在图片中找不到手链 (bracelet)，请换一张图片试试。")

    # 3. 处理遮罩
    bracelet_mask = segmentation_result[0]['mask']
    background_mask = 1 - bracelet_mask
    
    # 4. 优化遮罩 (确保是二值遮罩)
    processed_mask = process_mask(background_mask)
    
    # 5. 增强提示词
    enhanced_prompt = enhance_prompt(background_prompt)

    # 6. 执行图像修补替换背景
    result_image = flux_image_inpainting(
        prompt=enhanced_prompt,
        image=image,
        mask=processed_mask
    )

    return result_image

def main():
    st.title("手链背景替换示范")
    st.write("上传含有手链的图片，并尝试将背景替换成「简单又有质感」的风格。")

    # 提供几种预设 Prompt（改进的提示词）
    style_prompts = {
        "极简白底风格": "pure white background with subtle gradient lighting, professional studio setup, clean and minimal",
        "奶油色调高级纹理": "luxurious cream colored background with soft natural lighting, subtle marble texture, premium feel",
        "霧面灰色現代感": "sophisticated matte gray background with depth, professional product photography lighting, modern aesthetic",
        "溫和淺色布紋": "delicate light fabric texture background with natural shadows, soft studio lighting, gentle folds",
        "質感紙張效果": "premium textured paper background with natural grain, soft diffused lighting, artistic composition",
        "自订": ""
    }

    style_choice = st.selectbox("選擇背景風格", list(style_prompts.keys()))

    if style_choice == "自訂":
        background_prompt = st.text_input(
            "請輸入自訂背景描述",
            value="a minimalistic white background with soft shadows, high-quality and clean, subtly elegant"
        )
    else:
        # 使用預設 prompt
        background_prompt = style_prompts[style_choice]

    # 檔案上傳器
    uploaded_file = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg", "heic"])

    if uploaded_file is not None:
        # 將上傳檔案暫存
        temp_input_path = "temp_input_image"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 顯示原始圖片
        st.image(temp_input_path, caption="原始圖片", use_container_width=True)

        # 按鈕：執行背景替換
        if st.button("開始替換背景"):
            try:
                # 進行背景替換
                result_image = segment_and_replace_background(
                    image_path=temp_input_path,
                    background_prompt=background_prompt
                )

                # 存檔與顯示
                output_path = "bracelet_with_background.png"
                save_image(result_image, output_path)
                st.image(output_path, caption="替換後結果", use_container_width=True)
                st.success("背景替換完成！")

            except Exception as e:
                st.error(f"執行失敗：{e}")


if __name__ == "__main__":
    main()
