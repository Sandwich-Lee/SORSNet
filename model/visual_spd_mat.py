from matplotlib import pyplot as plt


def visualize_spd_matrix(spd_mat, layer_name, save_path="./spd_vis", vmin=None, vmax=None):
    """
    可视化SPD矩阵（热力图）
    Args:
        spd_mat: SPD矩阵张量，shape=[B, D, D]（B=批次，D=矩阵维度/通道数）
        layer_name: 层名称（用于命名保存文件）
        save_path: 保存路径
        vmin/vmax: 热力图数值范围（None则自动适配）
    """
    # 创建保存目录
    import os
    os.makedirs(save_path, exist_ok=True)

    # 遍历批次中的每个样本（通常取第0个样本可视化）
    for batch_idx in range(min(spd_mat.shape[0], 1)):  # 只可视化第一个样本
        mat = spd_mat[batch_idx].numpy()  # [D, D]

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        # 使用RdBu_r配色（红=高值，蓝=低值，白色=中间值），也可换viridis/rainbow
        im = plt.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(im, shrink=0.8, label="Covariance Value")
        plt.title(f"SPD Matrix - {layer_name} (Batch {batch_idx})", fontsize=14)
        plt.xlabel("Channel Index", fontsize=12)
        plt.ylabel("Channel Index", fontsize=12)

        # 可选：添加数值标注（矩阵维度较小时启用，如D≤32）
        # if mat.shape[0] <= 32:
        #     for i in range(mat.shape[0]):
        #         for j in range(mat.shape[1]):
        #             plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6, color="black")

        # 保存图片
        save_file = os.path.join(save_path, f"{layer_name}_batch{batch_idx}.png")
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SPD矩阵可视化结果已保存：{save_file}")