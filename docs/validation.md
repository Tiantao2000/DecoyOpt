# 验证记录

日期：2026-10-28。复用 `arg5080` 环境，没有创建环境或安装依赖。

- 物理 GPU 1（NVIDIA A800 80GB），通过 `CUDA_VISIBLE_DEVICES=1` 隔离。
- 54 条真实序列：12 条酶、6 条未注释蛋白及 36 条 decoy，分布在训练、验证和测试集。
- 从本机已有的 ProtT5-XL-UniRef50 模型生成 1024 维缓存；重复命令跳过已完成缓存，矩阵哈希保持一致。
- 二分类与 EC 模型各训练 2 轮，并完成测试。另一次短训将 EC 刷新周期设为 1，覆盖负样本池更新和 Adam 重置。
- 使用提供的 40/fold_1 权重完成小样本测试，并在 GPU 上加载全部 15 组、30 个模型检查有限输出。
- 同一小样本及缓存下，重构代码与原 CLEAN_OPT 的测试 logit 逐元素完全相同，EC 候选标签集合完全相同。
- 三个指标回归检查通过：多标签指标对照 scikit-learn、拒绝酶的 FN 计数、含并列分数的验证 MCC 阈值选择。
- 五折汇总流程通过，使用复制的小样本结果检查 SD；这些结果只用于功能验证。
- 完整 15 个折逐一对照原始加载器：每折 597,448 条记录，ID、标签、来源、划分和 P/D0/D5/D10 分组全部一致。
- 数据、权重 manifest 中所有文件哈希通过；两个资源压缩包逐文件内容校验通过。

测试序列、缓存、短训权重、预测、汇总及临时目录均已删除。保留的 `data/`、
`checkpoints/` 和 `dist/` 是正式交付资源。未重跑全量训练或全量性能评估。

复现小规模 GPU 测试（使用现有环境及已有本地模型）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python tests/smoke_gpu.py --model /path/to/prot_t5_xl_uniref50 \
  --source /path/to/enzyme_classification
```
