from vision_agent.tools import (
    load_image,
    florence2_sam2_instance_segmentation,
    flux_image_inpainting,
    save_image
)
import streamlit as st
import os
from typing import *
import numpy as np

# Pillow HEIF 相關
from pillow_heif import register_heif_opener
register_heif_opener()

# Vision Agent

def segment_and_replace_background(
    image_path: str,
    background_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    output_path: str = "bracelet_with_background.png"
) -> str:
    """
    1. 讀取圖片
    2. 用 florence2_sam2_instance_segmentation 偵測手鍊
    3. 取得手鍊遮罩並反轉得到背景遮罩
    4. 用 flux_image_inpainting 以自訂的背景風格替換該區域
    5. 儲存並回傳新圖檔路徑
    """

    # 1. 載入圖片
    image = load_image(image_path)

    # 2. 偵測手鍊
    segmentation_result = florence2_sam2_instance_segmentation("bracelet", image)
    if not segmentation_result:
        raise ValueError("在圖片中找不到手鍊 (bracelet)，請換一張圖片試試。")

    # 3. 二值化遮罩 (hand/bracelet)
    bracelet_mask = segmentation_result[0]['mask']

    # 4. 反轉遮罩得到背景部分
    background_mask = 1 - bracelet_mask

    # 5. 執行影像修補替換背景
    #   假設 flux_image_inpainting 接受以下參數：
    #   - prompt (str)
    #   - init_image (PIL.Image 或 np.ndarray)
    #   - mask (np.ndarray)
    #   - num_inference_steps (int) 可選
    #   - guidance_scale (float) 可選
    #   若版本不支援，請自行調整或刪除參數。
    result_image = flux_image_inpainting(
        prompt=background_prompt,
        init_image=image,
        mask=background_mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # 6. 儲存結果圖片
    save_image(result_image, output_path)

    # 7. 回傳結果路徑
    return output_path


def main():
    st.title("手鍊背景替換示範")
    st.write("上傳含有手鍊的圖片，並嘗試將背景替換成不同風格。")

    # 風格選單
    style_options = ["絲綢", "木質", "金屬", "自訂"]
    style_choice = st.selectbox("選擇背景風格", style_options)

    # 預設絲綢的 prompt
    silk_prompt = "high-quality silk fabric texture, smooth and shiny, elegant background"

    # 依照選擇改變 prompt
    if style_choice == "絲綢":
        background_prompt = silk_prompt
    elif style_choice == "木質":
        background_prompt = "wooden texture background, warm and rustic, high-quality, detailed"
    elif style_choice == "金屬":
        background_prompt = "metallic background with brushed steel, futuristic, high-quality, detailed"
    else:
        # 讓使用者自訂
        background_prompt = st.text_input("請輸入自訂背景描述", value=silk_prompt)

    # 可以用 slider 或 number_input 來讓使用者控制生成品質
    num_steps = st.slider("生成步數 (越高越精細但速度越慢)", 20, 150, 50)
    guidance = st.slider("引導強度 (越高越貼近提示詞)", 1.0, 15.0, 7.5)

    # 檔案上傳器
    uploaded_file = st.file_uploader("上傳圖片", type=["png", "jpg", "jpeg", "heic"])

    if uploaded_file is not None:
        # 將上傳檔案存在暫存路徑
        temp_input_path = "temp_input_image"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 顯示原始圖片
        st.image(temp_input_path, caption="原始圖片", use_container_width=True)

        # 按鈕：執行背景替換
        if st.button("開始替換背景"):
            try:
                output_path = segment_and_replace_background(
                    image_path=temp_input_path,
                    background_prompt=background_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance
                )
                # 顯示結果圖片
                st.image(output_path, caption="替換後結果", use_container_width=True)
                st.success("背景替換完成！")
            except Exception as e:
                st.error(f"執行失敗：{e}")


if __name__ == "__main__":
    main()
