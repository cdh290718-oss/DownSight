# 运行报告

- 状态：用户确认后完成目录整理；推送前检查见 [证据](evidence/prepush.txt)。
- 基线：`db6f68e315e4cf1074e1f31d3041cdffc6656f5a`；原有未跟踪文件保留不动。
- 已创建 `calibration/downward_fisheye_camera/` 和设备标定说明；三份 JSON 与已有目录逐字节一致，见 [校验](evidence/verification.txt)。未进行推理或重新标定。
- 按用户提供的协作规范，修改或删除非本轮生成文件须用户回复“确认”。用户确认的操作清单：删除上一轮创建的 `calibration/session_hand6_head01_100/` 中四个文件及空目录（已在新目录保存相同的三个 JSON 和新版说明），并将上一轮 `runs/2026-11-07_session-hand6-calibration/report.md` 中的标定目录链接更新为新路径，以免链接失效。保留历史 brief 和证据，作为历史记录；历史 Git 提交也不改写。随后进行发布前自检、提交并推送。
- 理由：移除旧目录中的旧名称及描述，让仓库使用设备标定名称，并保持报告中的目录链接有效。
