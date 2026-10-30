# 发布文件

将以下两个文件上传到 Google Drive，并开启链接共享：

- `dist/decoyopt_data.tar.gz`：三种划分 × 五折数据、完整序列及 decoy 亲本关系。
- `dist/decoyopt_checkpoints.tar.gz`：15 组 ProtT5-XL-UniRef50 模型，每组包含二分类与 EC 权重、中心及词表。

上传后，将 README 中的 `DATA_FILE_ID` 和 `CHECKPOINT_FILE_ID` 替换为真实文件 ID。
Google Drive 链接 `https://drive.google.com/file/d/FILE_ID/view` 中的 `FILE_ID`
就是下载命令所需的值。发布 `dist/SHA256SUMS.txt` 可供下载后核验。

`dist/decoyopt_source.tar.gz` 是 GitHub 源码包；解压后上传其中的文件。
数据、模型、缓存及运行结果已在 `.gitignore` 中排除。

在空输出目录中重新打包原始资源：

```bash
python tools/package_assets.py --source /path/to/enzyme_classification --output /path/to/output
```

数据包只保留运行所需的序列、标签和划分；原始样本顺序、EC 多标签、固定测试集及
decoy 分组保持一致。权重文件与原始 CLEAN_OPT 文件逐字节一致，包内 manifest
记录文件 SHA-256。`reference_metrics.json` 保留原始完整测试集结果。
