import streamlit as st
import os
import numpy as np
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

def segment_and_replace_background(
    image_path: str,
    background_prompt: str,
    output_path: str = "bracelet_with_background.png",
) -> np.ndarray:
    """
    1. 讀取圖片
    2. 用 florence2_sam2_instance_segmentation 偵測手鍊
    3. 取得手鍊遮罩並反轉得到背景遮罩
    4. 用 flux_image_inpainting 以自訂的背景風格替換該區域
    5. 回傳 (不直接存檔) 生成的影像 (np.ndarray 或 PIL Image)
    """
    # 1. 載入圖片
    image = load_image(image_path)

    # 2. 偵測手鍊
    segmentation_result = florence2_sam2_instance_segmentation("bracelet", image)
    if not segmentation_result:
        raise ValueError("在圖片中找不到手鍊 (bracelet)，請換一張圖片試試。")

    # 3. 二值化遮罩 (bracelet)
    bracelet_mask = segmentation_result[0]['mask']

    # 4. 反轉遮罩得到背景部分
    background_mask = 1 - bracelet_mask

    # 5. 執行影像修補替換背景
    result_image = flux_image_inpainting(
        prompt=background_prompt,
        image=image,
        mask=background_mask
    )

    return result_image


def main():
    st.title("手鍊背景替換示範")
    st.write("上傳含有手鍊的圖片，並嘗試將背景替換成「簡單又有質感」的風格。")

    # 提供幾種預設 Prompt
    style_prompts = {
        "極簡白底風格": "a minimalistic white background with soft shadows, high-quality and clean, subtly elegant",
        "奶油色調高級紋理": "a warm creamy texture background, luxurious and high-quality, soft lighting, subtle gradients",
        "霧面灰色現代感": "a smooth matte gray background, modern, high-quality, subtle depth, elegant minimal design",
        "溫和淺色布紋": "a light pastel fabric texture, soft and cozy, subtle details, high-quality minimalistic design",
        "質感紙張效果": "a high-quality paper texture background, slightly off-white, subtle grain, minimalist and elegant",
        "自訂": ""
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
