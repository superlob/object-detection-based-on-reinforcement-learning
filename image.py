import cv2
import os

def capture_images(num_images=100):
    # 创建保存图像的目录
    save_dir = os.path.join("collected_images")
    os.makedirs(save_dir, exist_ok=True)

    # 打开摄像头（默认摄像头，索引通常是0）
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: 无法访问摄像头")
        return

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: 无法读取摄像头画面")
            break

        # 显示实时画面
        cv2.imshow("Camera - Press SPACE to Capture", frame)

        # 检测按键（空格键触发保存）
        key = cv2.waitKey(1)
        if key == 32:  # 空格键ASCII码
            # 保存图像
            image_path = os.path.join(save_dir, f"{20+ count + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"已保存: {image_path}")
            count += 1

        elif key == 27:  # ESC键退出
            print("用户中断采集")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"采集完成！共保存 {count} 张图像到目录: {save_dir}")


if __name__ == "__main__":
    capture_images(num_images=100)