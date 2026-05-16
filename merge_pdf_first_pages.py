import os
import random
import glob
from PIL import Image
from pdf2image import convert_from_path

def merge_random_pdf_first_pages(folder_path, output_image_path, count=8):
    # 1. 获取目录下所有的 PDF 文件（包括子目录）
    pdf_files = glob.glob("**/*.pdf", root_dir=folder_path, recursive=True)
    
    # 检查 PDF 文件数量是否足够
    if not pdf_files:
        print(f"错误：在 '{folder_path}' 及其子目录下没有找到任何 PDF 文件。")
        return

    if len(pdf_files) < count:
        print(f"警告：目录下只有 {len(pdf_files)} 个 PDF 文件，将全部进行拼接。")
        selected_pdfs = pdf_files
    else:
        # 2. 随机抽取指定数量的 PDF
        selected_pdfs = random.sample(pdf_files, count)
    
    print("选中的 PDF 文件：")
    for idx, name in enumerate(selected_pdfs, 1):
        print(f"[{idx}] {name}")

    first_pages = []
    
    # 3. 读取每个 PDF 的第一页并转为 Image 对象
    for pdf in selected_pdfs:
        pdf_full_path = os.path.join(folder_path, pdf)
        try:
            # convert_from_path 会返回一个包含 PIL Image 对象的列表
            # first_page=1, last_page=1 表示只渲染第一页，dpi=150 兼顾清晰度和速度
            images = convert_from_path(pdf_full_path, first_page=1, last_page=1, dpi=150)
            if images:
                first_pages.append(images[0])
        except Exception as e:
            print(f"读取文件失败 {pdf}: {e}")

    if not first_pages:
        print("没有成功读取到任何 PDF 页面。")
        return

    # 4. 计算拼接后的长图尺寸
    # 为了保证拼接美观，我们将所有图片缩放到相同的宽度（以第一张图的宽度为准）
    target_width = first_pages[0].width
    
    # 统一调整所有图片宽度，并计算总高度
    resized_images = []
    total_height = 0
    for img in first_pages:
        if img.width != target_width:
            # 按比例缩放高度
            scale = target_width / img.width
            new_height = int(img.height * scale)
            img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
        total_height += img.height

    # 5. 创建一张空白的大画布
    result_image = Image.new('RGB', (target_width, total_height), color=(255, 255, 255))

    # 6. 竖直方向依次粘贴
    current_y = 0
    for img in resized_images:
        result_image.paste(img, (0, current_y))
        current_y += img.height

    # 7. 保存最终结果
    result_image.save(output_image_path)
    print(f"\n🎉 拼接完成！大图已保存至: {output_image_path}")

# --- 运行示例 ---
if __name__ == "__main__":
    # 替换为你的 PDF 文件夹路径
    target_folder = "./vis_out_64f_4curve_organized/short" 
    # 输出的长图名称
    output_result = target_folder + "/merged_short.jpg" 
    
    # 如果文件夹不存在，记得先创建或修改路径
    if os.path.exists(target_folder):
        merge_random_pdf_first_pages(target_folder, output_result, count=8)
    else:
        print(f"请先将 'target_folder' 修改为你想处理的真实目录路径。")