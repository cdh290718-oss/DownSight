# 运行报告

- 状态：用户确认后删除重复目录并更新链接；发布前检查见 [证据](evidence/prepush.txt)。
- 基线：`ccdcebb3acad4ed7cabad15d9ea6e4882c180a5e`，与开始时远端一致；保留已有未跟踪文件。
- 检查：`data/calibration/` 中的三份 JSON 与旧目录副本的 SHA-256 一致，见 [证据](evidence/verification.txt)。
- 已确认的操作清单：删除 `calibration/downward_fisheye_camera/` 内的 `README.md`、`cam3_left_lower.egosmplx.json`、`cam4_right_lower.egosmplx.json`、`stereoParams_20260604_fixed_native.json`，并删除两个空目录 `calibration/downward_fisheye_camera/` 和 `calibration/`；将 `runs/2026-11-07_session-hand6-calibration/report.md` 中的标定目录链接改为 `data/calibration/README.md`，避免链接失效。保留历史报告文本和证据的历史路径，不改写已推送历史。随后进行发布前自检、提交并推送到 GitHub。
- 理由：删除重复的根目录，同时保留 `data/calibration/` 下的标定文件及可用链接。
