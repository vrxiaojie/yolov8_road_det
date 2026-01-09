import json
import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 想要转换多少张图片？
MAX_IMAGES = 1000  # <--- 在这里修改你想要的数量

# 2. 原始路径设置
# 标注 JSON 文件路径
json_file_path = r'D:\BDD100K\bdd100k_labels_images_val.json'
# 原始图片所在的文件夹路径
source_image_dir = r'D:\BDD100K\bdd100k\images\100k\val'
# 类型 train val test（输出的子文件夹名称）
cls = 'val'

# 3. 输出路径设置 (建议新建一个干净的文件夹存放这个子集)
# 脚本会自动创建 images 和 labels 子文件夹
output_root_dir = 'datasets/bdd100k_mini/'

# 4. 类别映射 (保持不变)
target_classes = {
    'car': 0,
    'bus': 1,
    'person': 2,
    'truck': 3,
    'traffic sign': 4,
    'traffic light': 5
}

# BDD100K 标准尺寸
IMG_WIDTH = 1280
IMG_HEIGHT = 720


# ===========================================

def convert_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    dw = 1. / img_w
    dh = 1. / img_h
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2.0
    y_center = y1 + h / 2.0
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return (x_center, y_center, w, h)


def main():
    # 1. 准备输出目录
    out_img_dir = os.path.join(output_root_dir, 'images', f'{cls}')
    out_lbl_dir = os.path.join(output_root_dir, 'labels', f'{cls}')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    # 2. 读取 JSON 并构建【索引字典】
    # 这一步很关键，把列表变成字典，查询速度从 O(N) 变成 O(1)
    print(f"Loading JSON file: {json_file_path} ...")
    with open(json_file_path, 'r') as f:
        data_list = json.load(f)

    print("Building index from JSON...")
    # 结构变成: { 'xxx.jpg': [label1, label2...], 'yyy.jpg': [...] }
    labels_map = {item['name']: item.get('labels', []) for item in data_list}
    print(f"JSON index built. Total annotated images in JSON: {len(labels_map)}")
    # 3. 扫描本地图片文件夹
    print(f"Scanning image directory: {source_image_dir} ...")
    all_image_files = [f for f in os.listdir(source_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(all_image_files)} images in folder.")
    # 4. 截取需要处理的数量
    if MAX_IMAGES is not None:
        images_to_process = all_image_files[:min(MAX_IMAGES, len(all_image_files))]
    else:
        images_to_process = all_image_files

    print(f"Processing {len(images_to_process)} images...")
    count_success = 0
    count_no_annotation = 0
    # 5. 遍历本地图片，去 JSON 字典里找标签
    for img_filename in tqdm(images_to_process):

        # 尝试从字典里获取标签
        if img_filename not in labels_map:
            # 只有当你想知道哪些图没标签时才打开这行
            # print(f"Warning: No annotation found for {img_filename} in JSON.")
            count_no_annotation += 1
            continue
        labels = labels_map[img_filename]

        # --- 生成 YOLO 标签 ---
        valid_labels = []
        for label in labels:
            category = label['category']
            if category not in target_classes: continue
            if 'box2d' not in label: continue
            box2d = label['box2d']
            x1 = max(0, box2d['x1'])
            y1 = max(0, box2d['y1'])
            x2 = min(IMG_WIDTH, box2d['x2'])
            y2 = min(IMG_HEIGHT, box2d['y2'])

            cls_id = target_classes[category]
            yolo_box = convert_box((x1, y1, x2, y2), IMG_WIDTH, IMG_HEIGHT)
            valid_labels.append(f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
        # --- 保存 ---
        if valid_labels:
            # 写入 txt
            txt_filename = os.path.splitext(img_filename)[0] + '.txt'
            txt_path = os.path.join(out_lbl_dir, txt_filename)
            with open(txt_path, 'w') as f_out:
                f_out.write('\n'.join(valid_labels))

            # 复制图片 (为了构建独立的训练集)
            src_img_path = os.path.join(source_image_dir, img_filename)
            dst_img_path = os.path.join(out_img_dir, img_filename)
            shutil.copy2(src_img_path, dst_img_path)

            count_success += 1
    print("-" * 30)
    print(f"Processing Done!")
    print(f"Images processed: {len(images_to_process)}")
    print(f"Success (saved): {count_success}")
    print(f"No annotation/Skipped: {count_no_annotation}")
    print(f"Output saved to: {output_root_dir}")

if __name__ == "__main__":
    main()
