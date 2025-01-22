from vision_agent.tools import (
    load_image,
    florence2_sam2_instance_segmentation,
    flux_image_inpainting,
    save_image
)
import vision_agent as va
import streamlit as st
import os
from typing import *
import numpy as np

# Pillow HEIF 相關
from pillow_heif import register_heif_opener
register_heif_opener()

# Vision Agent


def segment_and_replace_background(image_path: str) -> str:
    """
    1. 讀取圖片
    2. 用 florence2_sam2_instance_segmentation 偵測手鍊
    3. 取得手鍊遮罩並反轉得到背景遮罩
    4. 用 flux_image_inpainting 以絲綢背景替換該區域
    5. 儲存並回傳新圖檔路徑
    """
    # 1. 載入圖片
    image = load_image(image_path)

    # 2. 偵測手鍊
    segmentation_result = florence2_sam2_instance_segmentation(
        "bracelet", image)
    if not segmentation_result:
        raise ValueError("在圖片中找不到手鍊 (bracelet)，請換一張圖片試試。")

    # 3. 二值化遮罩
    bracelet_mask = segmentation_result[0]['mask']

    # 4. 反轉遮罩得到背景部分
    background_mask = 1 - bracelet_mask

    # 5. 準備絲綢背景提示詞
    silk_prompt = "high-quality silk fabric texture, smooth and shiny, elegant background"

    # 6. 執行影像修補替換背景
    result_image = flux_image_inpainting(silk_prompt, image, background_mask)

    # 7. 儲存結果圖片
    output_path = "bracelet_with_silk_background.png"
    save_image(result_image, output_path)

    # 8. 回傳結果路徑
    return output_path


def main():
    st.title("手鍊背景替換為絲綢示範")
    st.write("上傳含有手鍊的圖片，並嘗試將背景替換成絲綢風格。")

    # 檔案上傳器
    uploaded_file = st.file_uploader(
        "上傳圖片", type=["png", "jpg", "jpeg", "heic"])

    if uploaded_file is not None:
        # 將上傳檔案存在暫存路徑
        temp_input_path = "temp_input_image"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 顯示原始圖片
        st.image(temp_input_path, caption="原始圖片", use_column_width=True)

        # 按鈕：執行背景替換
        if st.button("開始替換背景"):
            try:
                output_path = segment_and_replace_background(temp_input_path)
                # 顯示結果圖片
                st.image(output_path, caption="替換後結果", use_column_width=True)
                st.success("背景替換完成！")
            except Exception as e:
                st.error(f"執行失敗：{e}")


if __name__ == "__main__":
    main()
