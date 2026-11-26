# 自查

- evidence/check_arm_direct_sanitized.txt 对应四行 2D 指标和四个 3D 余量；exit_status_direct.txt 为 Python 退出码 0。
- 简化对象的字段对照 paired.body 与 fit_probe.Frame 初始化：neutral layer、global0 和零 beta；绕开分割输入，未拟合新姿态。
- 最大伸展距离只是必要条件；未验证关节角、碰撞或双目可行性。003 左臂余量仅 0.50 cm。
- 原腕三维位置与二维检测目标不一定一致；腕对齐指尖误差也不是单独的手形度量。
- 原始日志只保留在开发机；待发布日志已将开发机路径替换为占位符。
