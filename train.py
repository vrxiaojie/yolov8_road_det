import argparse
import datetime
from ultralytics import YOLO


def main(opt):
    # 1. 训练名称
    train_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. 加载模型
    print(f"正在加载模型配置: {opt.cfg}")
    model = YOLO(opt.cfg)
    print(f"训练参数: Epochs={opt.epochs}, Batch={opt.batch}, Imgsz={opt.imgsz}, Data={opt.data}")

    # 3. 开始训练
    # 动态构建 run 的名称
    run_name = f"{train_name}_{opt.imgsz}_{opt.cfg.split('.yaml')[0]}"

    model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        workers=opt.workers,
        name=run_name,
        device=opt.device,  # 添加了设备选择，方便切换 CPU/GPU
        resume=opt.resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')

    # 添加命令行参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--data', type=str, default='data.yml', help='数据集配置文件路径')
    parser.add_argument('--batch', type=float, default=0.8,
                        help='Batch size (可以是整数或 0.0-1.0 之间的浮点数表示显存占比)')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--workers', type=int, default=0, help='Dataloader workers (Windows下建议设为0)')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, default='cfg/yolov8n.yaml', help='模型配置文件路径')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    opt = parser.parse_args()

    main(opt)
