import streamlit as st
import os
import numpy as np
from typing import *
from pillow_heif import register_heif_opener
import replicate
from PIL import Image
import io
register_heif_opener()

# Vision Agent 套件
from vision_agent.tools import (
    load_image,
    florence2_sam2_instance_segmentation,
    save_image
)

def image_to_bytes(image_array):
    """将numpy数组转换为bytes"""
    image = Image.fromarray(image_array)
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    return byte_stream.getvalue()

def segment_and_replace_background(
    image_path: str,
    background_prompt: str,
    output_path: str = "bracelet_with_background.png",
) -> np.ndarray:
    """
    使用 Replicate 的 SDXL 模型进行背景替换
    """
    # 1. 载入图片
    image = load_image(image_path)

    # 2. 检测手链
    segmentation_result = florence2_sam2_instance_segmentation("bracelet", image)
    if not segmentation_result:
        raise ValueError("在图片中找不到手链 (bracelet)，请换一张图片试试。")

    # 3. 获取遮罩
    bracelet_mask = segmentation_result[0]['mask']
    background_mask = 1 - bracelet_mask
    
    # 4. 准备图片和遮罩
    image_bytes = image_to_bytes(image)
    mask_bytes = image_to_bytes((background_mask * 255).astype(np.uint8))

    # 5. 调用 Replicate API 进行图像修补
    output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "image": image_bytes,
            "mask": mask_bytes,
            "prompt": background_prompt,
            "negative_prompt": "ugly, blurry, low quality, distorted, deformed",
            "num_outputs": 1,
            "scheduler": "K_EULER",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "seed": 42,
        }
    )
    
    # 6. 下载生成的图片
    if output and len(output) > 0:
        result_image = load_image(output[0])
        return result_image
    else:
        raise Exception("图像生成失败")

def main():
    st.title("手链背景替换示范")
    st.write("上传含有手链的图片，并尝试将背景替换成「简单又有质感」的风格。")

    # 检查是否设置了 REPLICATE_API_TOKEN
    if not os.environ.get("REPLICATE_API_TOKEN"):
        st.error("请设置 REPLICATE_API_TOKEN 环境变量")
        return

    # 提供几种预设 Prompt
    style_prompts = {
        "极简白底风格": "professional product photo with pure white background, soft shadows, studio lighting setup, clean and minimal",
        "奶油色调高级纹理": "professional product photo with luxurious cream colored background, soft natural lighting, subtle marble texture, premium feel",
        "霧面灰色現代感": "professional product photo with sophisticated matte gray background, studio lighting, modern aesthetic",
        "溫和淺色布紋": "professional product photo with delicate light fabric texture background, soft studio lighting, gentle folds",
        "質感紙張效果": "professional product photo with premium textured paper background, natural grain, artistic composition",
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
