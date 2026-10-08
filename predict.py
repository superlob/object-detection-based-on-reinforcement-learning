from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


image_paths = [f"test_exp/{i-2}.JPG" for i in range(12)]  # 替换为实际图片路径
images = [np.array(Image.open(path)) for path in image_paths]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
img_display = ax.imshow(images[0])

def update(frame):
    img_display.set_array(images[frame])
    ax.set_title(f"Frame: {frame+1}/{len(images)}", fontsize=12)
    return [img_display]

ani = FuncAnimation(
    fig,
    update,
    frames=len(images),
    interval=1000,  # 帧间隔(ms)
    blit=True      # 优化渲染性能
)

ani.save("animation.mp4", writer='pillow', fps=1, dpi=100)
plt.tight_layout()
plt.show()