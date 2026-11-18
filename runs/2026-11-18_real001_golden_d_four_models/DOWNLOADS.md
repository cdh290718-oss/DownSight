# SMPL-X 参数下载

[一次下载全部参数 ZIP](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0_smplx_new4_and_legacy_all.zip)

共 11 份拟合参数；ZIP 含 NPZ、同值 JSON、相机标定、cam3→cam4 变换、说明及 SHA-256 清单。NPZ 只含拟合系数与小型元数据，不含模型权重或完整原始输出。

| 分类 | 说明 | 坐标系 | 下载 |
|---|---|---|---|
| 01_new_four_models | Current full-2D golden D, detector original, shared body for cam3/cam4 | cam3 CV | [real001_t029p0__goldenD2d__original__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__goldenD2d__original__shared_cam3_cv.npz) |
| 01_new_four_models | Current full-2D golden D, detector rw500, shared body for cam3/cam4 | cam3 CV | [real001_t029p0__goldenD2d__rw500__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__goldenD2d__rw500__shared_cam3_cv.npz) |
| 01_new_four_models | Current full-2D golden D, detector rw5000, shared body for cam3/cam4 | cam3 CV | [real001_t029p0__goldenD2d__rw5000__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__goldenD2d__rw5000__shared_cam3_cv.npz) |
| 01_new_four_models | Current full-2D golden D, detector rw20000, shared body for cam3/cam4 | cam3 CV | [real001_t029p0__goldenD2d__rw20000__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__goldenD2d__rw20000__shared_cam3_cv.npz) |
| 02_legacy_three_methods | Archived original per-view SMPL-X body coefficients; not final MANO seam mesh | cam3 CV | [real001_t029p0__legacy_original__cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_original__cam3_cv.npz) |
| 02_legacy_three_methods | Archived v2 per-view SMPL-X body coefficients; not final MANO seam mesh | cam3 CV | [real001_t029p0__legacy_v2__cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_v2__cam3_cv.npz) |
| 02_legacy_three_methods | Archived original per-view SMPL-X body coefficients; not final MANO seam mesh | cam4 CV | [real001_t029p0__legacy_original__cam4_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_original__cam4_cv.npz) |
| 02_legacy_three_methods | Archived v2 per-view SMPL-X body coefficients; not final MANO seam mesh | cam4 CV | [real001_t029p0__legacy_v2__cam4_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_v2__cam4_cv.npz) |
| 02_legacy_three_methods | Archived old 3D D: baseline, shared cam3/cam4 body | cam3 CV | [real001_t029p0__legacy_D3d__baseline__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_D3d__baseline__shared_cam3_cv.npz) |
| 03_legacy_head_perturbations | Archived old 3D D: head_tx_plus20mm_ry_plus3deg, shared cam3/cam4 body | cam3 CV | [real001_t029p0__legacy_D3d__head_tx_plus20mm_ry_plus3deg__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_D3d__head_tx_plus20mm_ry_plus3deg__shared_cam3_cv.npz) |
| 03_legacy_head_perturbations | Archived old 3D D: head_tx_minus20mm_ry_minus3deg, shared cam3/cam4 body | cam3 CV | [real001_t029p0__legacy_D3d__head_tx_minus20mm_ry_minus3deg__shared_cam3_cv.npz](https://github.com/cdh290718-oss/DownSight/releases/download/real001-goldend2d-four-models-20261118/real001_t029p0__legacy_D3d__head_tx_minus20mm_ry_minus3deg__shared_cam3_cv.npz) |

`shared_cam3_cv` 是同一套双目共享人体参数，使用 cam3 CV 坐标；投影 cam4 时变换重建出的网格，不要把两个相机的参数相加。旧原方法/v2 是每目独立拟合。

旧 D 的 baseline、head_tx_plus20mm_ry_plus3deg、head_tx_minus20mm_ry_minus3deg 对应原报告的基础与正负联合锚点扰动三组。其源 NPZ 哈希与旧报告证据一致。

旧原方法/v2 匹配图包含 MANO 手部替换与接缝处理；这些额外网格编辑不由 SMPL-X 身体系数表达。

[参数格式、手型均值与加载说明](DOWNLOADS_README.md) · [逐文件 SHA-256 和重载检查](evidence/exports.json)
