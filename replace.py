import streamlit as st
import os
import numpy as np
from typing import *
from pillow_heif import register_heif_opener
from PIL import Image
import io
import torch
from diffusers import StableDiffusionInpaintPipeline
register_heif_opener()

# Vision Agent 套件
from vision_agent.tools import (
    load_image,
    florence2_sam2_instance_segmentation,
    save_image
)

@st.cache_resource
def load_inpainting_model():
    """加载 Stable Diffusion inpainting 模型"""
    model_id = "runwayml/stable-diffusion-inpainting"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False,
            token=st.secrets.get("HF_TOKEN", None),
        )
        
        # 如果是 CPU，转换为 float32
        if device == "cpu":
            pipe = pipe.to(torch.float32)
        
        pipe = pipe.to(device)
        return pipe
        
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
        raise e

def prepare_image_and_mask(image_array, mask_array):
    """准备图片和遮罩为 PIL Image 格式"""
    image = Image.fromarray(image_array)
    # 确保遮罩是黑白图片
    mask = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
    
    # 调整图片大小为 512x512（模型要求）
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    
    return image, mask

def segment_and_replace_background(
    image_path: str,
    background_prompt: str,
    output_path: str = "bracelet_with_background.png",
) -> np.ndarray:
    """
    使用 Stable Diffusion inpainting 模型进行背景替换
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
    pil_image, pil_mask = prepare_image_and_mask(image, background_mask)
    
    # 5. 加载模型
    pipe = load_inpainting_model()
    
    # 6. 生成图片
    with st.spinner('正在生成图片...'):
        output = pipe(
            prompt=background_prompt,
            negative_prompt="ugly, blurry, low quality, distorted, deformed",
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]
    
    # 7. 转换回 numpy array
    result_image = np.array(output)
    
    return result_image

def main():
    st.title("手链背景替换示范")
    st.write("上传含有手链的图片，并尝试将背景替换成「简单又有质感」的风格。")

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
