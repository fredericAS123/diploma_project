# StreamUAV-QA 样本审查报告

> 生成时间: 2026-03-02 18:56
> QA 总数: 59
> 视频数: 4

---

## 📊 数据集统计

| 维度 | 类别 | 数量 | 占比 |
|------|------|------|------|
| 类型 | backward | 19 | 32.2% |
| 类型 | forward | 3 | 5.1% |
| 类型 | realtime | 37 | 62.7% |
| 难度 | L1 | 37 | 62.7% |
| 难度 | L2 | 15 | 25.4% |
| 难度 | L3 | 7 | 11.9% |

## 🎬 视频信息

- **uav0000013_00000_v**: 269 帧, 8.97s, 尺寸 [1344, 756], 采样 9 帧, 事件 7 个
- **uav0000013_01073_v**: 58 帧, 1.93s, 尺寸 [1344, 756], 采样 2 帧, 事件 1 个
- **uav0000013_01392_v**: 118 帧, 3.93s, 尺寸 [1344, 756], 采样 4 帧, 事件 3 个
- **uav0000020_00406_v**: 501 帧, 16.7s, 尺寸 [1344, 756], 采样 17 帧, 事件 6 个
- **uav0000071_03240_v**: 181 帧, 6.03s, 尺寸 [1904, 1071], 采样 7 帧, 事件 5 个

## 🟢 L1 样本 (37 条)

### L1-1: [realtime] uav0000013_00000_v

**⏱ 时间戳**: 4.0s

**❓ 问题**: Is there any people visible in the current drone footage?

  ✅ A. Yes
     B. No

**📋 正确答案**: A. Yes

**🔍 依据**: `Frame@4.0s: people present (8)`

---

### L1-2: [realtime] uav0000013_00000_v

**⏱ 时间戳**: 1.0s

**❓ 问题**: How many motor(s) are currently visible in the aerial view?

     A. 5
     B. 4
  ✅ C. 7
     D. 3

**📋 正确答案**: C. 7

**🔍 依据**: `Frame@1.0s: motor=7`

---

### L1-3: [realtime] uav0000013_00000_v

**⏱ 时间戳**: 0.0s

**❓ 问题**: How many pedestrian(s) are currently visible in the aerial view?

     A. 8
     B. 5
     C. 7
  ✅ D. 10

**📋 正确答案**: D. 10

**🔍 依据**: `Frame@0.0s: pedestrian=10`

---

### L1-4: [realtime] uav0000013_00000_v

**⏱ 时间戳**: 4.0s

**❓ 问题**: How would you describe the current scene density from the aerial perspective?

     A. Very densely crowded
  ✅ B. Densely populated with many objects
     C. Sparse with only a few objects
     D. Moderately populated

**📋 正确答案**: B. Densely populated with many objects

**🔍 依据**: `Frame@4.0s: density=dense, total=31`

---

### L1-5: [realtime] uav0000013_00000_v

**⏱ 时间戳**: 5.0s

**❓ 问题**: What is the most common type of object currently visible from the drone?

     A. truck
  ✅ B. pedestrian
     C. people
     D. tricycle

**📋 正确答案**: B. pedestrian

**🔍 依据**: `Frame@5.0s: dominant=pedestrian, counts={'pedestrian': 15, 'motor': 7, 'people': 8, 'bicycle': 1, 'awning-tricycle': 1}`

---

## 🟡⭐ L2 样本 (15 条)

### L2-1: [backward] uav0000013_00000_v

**⏱ 时间戳**: 5.0s

**❓ 问题**: What type of object has recently appeared in the scene?

  ✅ A. awning-tricycle
     B. motor
     C. truck
     D. bicycle

**📋 正确答案**: A. awning-tricycle

**🔍 依据**: `awning-tricycle appeared in scene @5.0s`

---

### L2-2: [backward] uav0000013_00000_v

**⏱ 时间戳**: 8.0s

**❓ 问题**: What type of object has recently disappeared from the scene?

  ✅ A. awning-tricycle
     B. truck
     C. van
     D. tricycle

**📋 正确答案**: A. awning-tricycle

**🔍 依据**: `awning-tricycle disappeared from scene @8.0s`

---

### L2-3: [backward] uav0000013_00000_v

**⏱ 时间戳**: 2.0s

**❓ 问题**: Compared to earlier frames, has the total number of objects in the scene increased or decreased?

  ✅ A. Increased by about 2 (from 23 to 25)
     B. Decreased by about 2 (from 23 to 21)
     C. Remained roughly the same
     D. Cannot be determined from the video

**📋 正确答案**: A. Increased from 23 to 25

**🔍 依据**: `Count change: 23→25 over 1.0s`

---

### L2-4: [backward] uav0000013_00000_v

**⏱ 时间戳**: 6.0s

**❓ 问题**: How has the scene density changed over the recent video segment?

  ✅ A. Changed from dense to moderate
     B. Changed from moderate to dense
     C. Remained at dense level throughout
     D. Fluctuated unpredictably

**📋 正确答案**: A. Density changed from dense to moderate

**🔍 依据**: `Density: dense→moderate`

---

### L2-5: [backward] uav0000013_00000_v

**⏱ 时间戳**: 8.0s

**❓ 问题**: Based on your observation of the video stream, what is the overall trend of traffic flow?

     A. Traffic is gradually increasing
     B. Traffic is gradually decreasing
  ✅ C. Traffic has remained relatively stable
     D. Traffic is fluctuating unpredictably

**📋 正确答案**: C. Traffic is stable

**🔍 依据**: `Vehicle counts (recent half): [7, 8, 6, 5, 4]`

---

## 🔴 L3 样本 (7 条)

### L3-1: [backward] uav0000013_00000_v

**⏱ 时间戳**: 8.0s

**❓ 问题**: What is the correct chronological order of these observed events?

  ✅ A. pedestrian count increased from 10 to 12 → All 1 tricycle(s) left the scene → pedestrian count increased from 20 to 22
     B. pedestrian count increased from 20 to 22 → All 1 tricycle(s) left the scene → pedestrian count increased from 10 to 12
     C. pedestrian count increased from 10 to 12 → pedestrian count increased from 20 to 22 → All 1 tricycle(s) left the scene
     D. They occurred simultaneously

**📋 正确答案**: A. pedestrian count increased from 10 to 12 → All 1 tricycle(s) left the scene → pedestrian count increased from 20 to 22

**🔍 依据**: `Events sorted by time: [{"desc": "pedestrian count increased from 10 to 12", "ts": 2.0}, {"desc": "All 1 tricycle(s) left the scene", "ts": 4.0}, {"desc": "pedestrian count increased from 20 to 22", "ts": 8.0}]`

---

### L3-2: [forward] uav0000013_00000_v

**⏱ 时间戳**: 8.0s

**❓ 问题**: Based on the traffic trend observed, what is your prediction for the near future?

     A. Traffic congestion is likely to worsen
     B. Traffic congestion is likely to improve
  ✅ C. Traffic will likely remain at a similar level
     D. Insufficient information to make a prediction

**📋 正确答案**: C. Traffic is likely to remain similar

**🔍 依据**: `Recent 5 vehicle counts: [7, 8, 6, 5, 4], slope=-0.75`

---

### L3-3: [backward] uav0000013_01392_v

**⏱ 时间戳**: 3.0s

**❓ 问题**: What is the correct chronological order of these observed events?

  ✅ A. Total objects increased by 6 → motor count increased from 5 to 8 → 1 tricycle(s) appeared in scene
     B. 1 tricycle(s) appeared in scene → motor count increased from 5 to 8 → Total objects increased by 6
     C. Total objects increased by 6 → 1 tricycle(s) appeared in scene → motor count increased from 5 to 8
     D. They occurred simultaneously

**📋 正确答案**: A. Total objects increased by 6 → motor count increased from 5 to 8 → 1 tricycle(s) appeared in scene

**🔍 依据**: `Events sorted by time: [{"desc": "Total objects increased by 6", "ts": 2.0}, {"desc": "motor count increased from 5 to 8", "ts": 2.0}, {"desc": "1 tricycle(s) appeared in scene", "ts": 3.0}]`

---

### L3-4: [backward] uav0000020_00406_v

**⏱ 时间戳**: 12.0s

**❓ 问题**: What is the correct chronological order of these observed events?

  ✅ A. pedestrian count increased from 12 to 15 → All 1 bicycle(s) left the scene → Total objects decreased by 3
     B. Total objects decreased by 3 → All 1 bicycle(s) left the scene → pedestrian count increased from 12 to 15
     C. pedestrian count increased from 12 to 15 → Total objects decreased by 3 → All 1 bicycle(s) left the scene
     D. They occurred simultaneously

**📋 正确答案**: A. pedestrian count increased from 12 to 15 → All 1 bicycle(s) left the scene → Total objects decreased by 3

**🔍 依据**: `Events sorted by time: [{"desc": "pedestrian count increased from 12 to 15", "ts": 10.0}, {"desc": "All 1 bicycle(s) left the scene", "ts": 10.0}, {"desc": "Total objects decreased by 3", "ts": 12.0}]`

---

### L3-5: [forward] uav0000020_00406_v

**⏱ 时间戳**: 16.0s

**❓ 问题**: Based on the traffic trend observed, what is your prediction for the near future?

     A. Traffic congestion is likely to worsen
     B. Traffic congestion is likely to improve
  ✅ C. Traffic will likely remain at a similar level
     D. Insufficient information to make a prediction

**📋 正确答案**: C. Traffic is likely to remain similar

**🔍 依据**: `Recent 5 vehicle counts: [4, 2, 2, 2, 4], slope=0.00`

---

## ✅ 审查要点

请检查以下方面：

1. **问题是否清晰？** — 读者能否理解在问什么？
2. **正确答案是否确实正确？** — 对照 source_evidence 验证
3. **干扰选项是否合理？** — 不能太明显也不能太离谱
4. **L2 时序问题是否需要跨帧信息？** — 这是核心差异化指标
5. **时间戳是否合理？** — 不应超出视频时长

如果发现系统性问题（如某类 QA 全部有误），请记录问题类型，
后续可以针对性修复 qa_generator.py 中的对应方法。