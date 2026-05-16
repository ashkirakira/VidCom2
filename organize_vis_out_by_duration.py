import os
import shutil
from datasets import load_dataset

# ================= 配置路径 =================
# 1. 你本地存放视频文件的文件夹路径
SOURCE_DIR = "./vis_out_64f_4curve"  
# 2. 你希望将分类后的文件夹整理到哪个目录下
TARGET_DIR = "./vis_out_64f_4curve_organized"        
# ============================================

def organize_videos_by_duration():
    # 1. 加载 Hugging Face 上的 Video-MME 数据集
    print("正在加载 Video-MME 数据集...")
    # 如果你在国内网络环境遇到困难，可以尝试配置 HF 镜像源，或者提前下载好数据集
    dataset = load_dataset("lmms-lab/Video-MME", split="test") 
    
    # 2. 构建 videoID 到 duration 的映射字典
    print("正在构建视频时长映射表...")
    video_duration_map = {}
    for sample in dataset:
        v_id = sample.get("videoID")
        duration = sample.get("duration") # 或者根据具体字段调整，比如具体的秒数转换
        
        if v_id and duration:
            # 确保 duration 作为文件夹名称时没有非法字符，并转为字符串
            video_duration_map[str(v_id)] = str(duration).strip()

    # 3. 遍历本地文件夹中的文件
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 本地源路径 {SOURCE_DIR} 不存在，请检查配置！")
        return

    print("开始整理文件...")
    success_count = 0
    missing_count = 0

    for filename in os.listdir(SOURCE_DIR):
        source_file_path = os.path.join(SOURCE_DIR, filename)
        
        # 略过文件夹，只处理文件
        if os.path.isdir(source_file_path):
            continue
            
        # 提取文件名中的 videoID (去掉后缀，例如 "v_12345.mp4" -> "v_12345")
        videoID, _ = os.path.splitext(filename)
        
        # 4. 匹配数据集中的时长并移动
        if videoID in video_duration_map:
            duration_folder_name = video_duration_map[videoID]
            
            # 创建目标文件夹 (例如: ./organized_videos/Short (<=1m)/)
            dest_dir = os.path.join(TARGET_DIR, duration_folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            # 移动文件
            dest_file_path = os.path.join(dest_dir, filename)
            shutil.move(source_file_path, dest_file_path)
            
            print(f"已移动: {filename} -> 文件夹 [{duration_folder_name}]")
            success_count += 1
        else:
            print(f"未在数据集中找到对应的 id: {filename}")
            missing_count += 1

    print("\n完成！")
    print(f"成功分类移动了 {success_count} 个文件。")
    if missing_count > 0:
        print(f"有 {missing_count} 个文件在数据集中未匹配到 videoID。")

if __name__ == "__main__":
    organize_videos_by_duration()