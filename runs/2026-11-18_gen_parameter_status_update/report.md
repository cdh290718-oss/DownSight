# Gen 2D 表与参数来源补充

基线 30a1484，读取上一轮 brief/report/review；review 尚待审核。按用户明确要求修改最新报告，新增 2D 表和参数来源表，数值来自既有 summary 的 valid19，见 evidence/summary.json。相机标定来源和转换核查通过不等于物理精度独立验证；DFM Head 位置不等于真值；无 Head 朝向约束，网络朝向仅初始化。没有重跑实验或修改配置和原始结果；其他已有工作树修改保留。

推送前自检：仅本轮文档、精简数值证据及用户指定报告更新；无权重/数据集/图片/视频/NPY/NPZ/完整原始输出、凭据/.env、完整 provenance/patch；单文件小于 50 MB，路径脱敏。工作区、暂存、待推提交及远端逐项检查见 evidence/prepush.json；未审计历史。
