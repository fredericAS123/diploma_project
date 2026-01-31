"""
动态采样频率和时间编码优化迭代 Prompt

项目背景
=======
需要实现 Qwen2.5-VL 的动态采样频率功能，使得模型能够正确理解视频中的时间流逝。

关键需求
========
1. 采样频率精确度：
   ✓ 1fps：50秒视频应采样成50帧（±0帧误差）
   ✓ 2fps：50秒视频应采样成100帧（±0帧误差）
   ✓ 0.5fps：50秒视频应采样成25帧（±0帧误差）
   ✓ 帧数必须是 temporal_patch_size=2 的倍数（官方要求）

2. 时间编码精度：
   ✓ second_per_grid_t 必须精确计算：second_per_grid_t = video_duration / num_grids
   ✓ num_grids = num_sampled_frames / temporal_patch_size
   ✓ 相邻两帧的时间编码差值应为 1/(2*target_fps) 秒
     （因为2帧=1个grid，1个grid=second_per_grid_t秒）

3. 官方标准遵循：
   ✓ temporal_patch_size = 2（官方固定）
   ✓ tokens_per_second = 4（官方配置）
   ✓ temporal_position = grid_idx * second_per_grid_t * tokens_per_second
   ✓ 整个视频时长必须被完整覆盖（无时间空隙，误差<1秒）

测试验证清单
===========
对于每个采样频率测试，验证以下项目：

□ 帧数验证
  - 计算：expected_frames = int(video_duration * target_fps)
  - 对齐：expected_frames = (expected_frames // 2) * 2
  - 验证：sampled_frames == expected_frames ✅

□ second_per_grid_t 验证
  - 计算：num_grids = sampled_frames / 2
  - 预期：expected_spgt = video_duration / num_grids
  - 验证：abs(second_per_grid_t - expected_spgt) < 0.001 ✅

□ 帧间隔时间编码验证
  - 两帧时间差：second_per_grid_t = 1 / target_fps（对于1个grid = 2帧）
  - temporal position差：second_per_grid_t * 4
  - 验证：相邻grid的temporal position差应为 (1/target_fps) * 4 ✅

□ 时间覆盖验证
  - 覆盖时长：total_covered = num_grids * second_per_grid_t
  - 误差：abs(total_covered - video_duration) < 1.0秒 ✅

测试执行步骤
===========
1. 运行详细测试报告：
   ```bash
   python temporal_encoding/test_sampling_frequency_detailed.py
   ```

2. 检查生成的报告：
   - 输出文件：temporal_encoding/sampling_frequency_report.txt
   - 验证所有测试项都显示 ✅

3. 关键指标：
   - 帧数验证通过：3/3 ✅
   - 时间编码参数通过：3/3 ✅
   - 覆盖范围验证通过：3/3 ✅

代码改进检查清单
==============
以下代码文件需要保持或改进：

文件：temporal_encoding/model/video_sampler.py
================================================
□ class VideoSampler 初始化正确
  - target_fps: 目标采样频率
  - temporal_patch_size: 2（官方）
  - tokens_per_second: 4（官方）

□ sample_frames() 方法
  - 计算 target_frame_count = int(video_duration * target_fps)
  - 对齐到 temporal_patch_size 的倍数
  - 均匀采样：indices = np.linspace(0, total_frames-1, target_frame_count)
  - 返回 (sampled_frames, second_per_grid_t, metadata)

□ _align_to_temporal_patch() 方法
  - 公式：(frame_count // temporal_patch_size) * temporal_patch_size
  - 确保帧数是2的倍数

□ _calculate_second_per_grid_t() 方法
  - 公式：second_per_grid_t = video_duration / (num_sampled_frames // temporal_patch_size)
  - 必须精确计算（浮点精度要求高）

□ sample_from_timestamps() 方法
  - 支持非均匀采样（用于Star+Stream混合帧）
  - 按时间戳查找最近的帧

□ calculate_expected_temporal_positions() 函数
  - 模拟官方 get_rope_index 计算
  - 输出完整的 temporal position ID 序列

□ validate_time_encoding() 函数
  - 验证覆盖时长
  - 计算时间误差
  - 容忍度设置合理

文件：temporal_encoding/model/smart_frame_manager.py
====================================================
□ get_all_frames() 返回格式：(frames, timestamps, metadata)
  - 必须返回每帧的时间戳
  - 时间戳按升序排列

□ metadata 包含关键字段：
  - 'star_frames': Star Memory中的帧数
  - 'stream_frames': Stream Memory中的帧数
  - 'unique_frames': 去重后的帧数
  - 'time_span': 总时间跨度
  - 'min_timestamp', 'max_timestamp': 时间范围

文件：temporal_encoding/model/delayed_batch_inference.py
========================================================
□ _encode_all_frames() 使用 sample_from_timestamps()
  - 获取帧和时间戳
  - 调用 sampler.sample_from_timestamps()
  - 保存采样元数据到 self.last_sample_metadata

□ last_sample_metadata 包含：
  - 'original_frames': 原始帧数
  - 'sampled_frames': 采样后帧数
  - 'second_per_grid_t': 时间编码参数
  - 'temporal_grids': temporal grid数量
  - 'video_duration': 视频总时长

文件：temporal_encoding/test_sampling_frequency_detailed.py
==========================================================
□ DetailedSamplingReport 类
  - test_sampling_frequency() 方法覆盖所有验证项
  - generate_summary() 生成完整报告
  - save_report() 输出到txt文件

常见问题排查
===========
问题1：帧数不对齐到temporal_patch_size
  原因：遗漏了 _align_to_temporal_patch() 调用
  解决：确保 target_frame_count = _align_to_temporal_patch(target_frame_count)

问题2：second_per_grid_t 计算不精确
  原因：整数除法或浮点精度问题
  解决：确保用浮点除法，保留足够小数位数

问题3：时间编码覆盖不完整
  原因：video_duration 计算错误，或者没有正确处理端点
  解决：验证 video_duration 从最后一帧的时间戳减去第一帧的时间戳

问题4：帧间隔时间不匹配采样频率
  原因：second_per_grid_t 没有正确反映时间跨度
  解决：检查公式：second_per_grid_t = video_duration / num_grids

问题5：与官方实现不兼容
  原因：参数设置与官方不符
  解决：确保 temporal_patch_size=2, tokens_per_second=4

验证通过标准
===========
测试成功的标准输出应该看起来像：

📈 测试统计：
   • 总测试数：3
   • 帧数验证通过：3/3 ✅
   • 时间编码参数通过：3/3 ✅
   • 覆盖范围验证通过：3/3 ✅

✨ 关键成就：
   1. ✅ 动态采样频率实现正确
   2. ✅ second_per_grid_t 计算精确
   3. ✅ 时间编码参数匹配官方标准
   4. ✅ 整个视频时长得到正确覆盖

迭代策略
========
1. 第一轮：运行 test_sampling_frequency_detailed.py，检查基础功能
2. 第二轮：如果测试失败，根据失败项调整计算公式
3. 第三轮：与官方实现对标，确保兼容性
4. 第四轮：集成到 DelayedBatchInferenceEngine，进行端到端测试
5. 最终：生成详细报告，展示所有验证通过的证据

报告输出位置
===========
- 详细报告：temporal_encoding/sampling_frequency_report.txt
- 测试输出：temporal_encoding/test_delayed_batch_inference_output.txt

关键文档参考
===========
1. Qwen2.5-VL 官方时间编码：
   - temporal_position = grid_idx * second_per_grid_t * tokens_per_second
   - 其中 grid_idx 从 0 到 (num_temporal_grids - 1)

2. Flash-VStream 论文中关于 Star+Stream 的定义：
   - Star Memory：关键帧集合
   - Stream Memory：最近窗口帧集合

3. 动态采样的核心公式：
   - target_frames = video_duration * target_fps
   - second_per_grid_t = video_duration / (target_frames / temporal_patch_size)
   - frame_interval = 1 / target_fps（秒/帧）
"""

print(__doc__)
