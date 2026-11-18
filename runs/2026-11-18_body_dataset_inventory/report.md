# 开发机身体姿态数据盘点

用户澄清要寻找 Gen 以外的数据集。基线 09cf25d，上一轮 review 待审核；只读核查数据盘目录、代表性文件和 EgoPW 数据说明，未下载、解压、修改共享盘或开展模型评测。其他既有工作树改动保持原样。

| 数据集 | 本机已确认内容 | 用途和待核实事项 |
|---|---|---|
| EgoWholeBody | render_people_adanna 目录、压缩包、SMPL-X / labels 文件；一条动作含 img、depth、seg、camera_info 和 joint_info | 合成身体数据，可优先检查关节和相机；完整人数、划分及能否组成下视双目尚未核实。 |
| EgoPW | 多个人物目录、官方训练/测试列表、压缩包；抽查序列含 imgs、external_imgs、pseudo_gt.pkl | 真实图像与伪标注；外部视角不自动等同下视双目。 |
| SceneEgo | 已发现图像、annotation.pkl、local_pose_gt.pkl 和训练/测试压缩包 | 有身体姿态评测接入代码；当前样本具体标注、相机与划分需进一步检查。 |
| UnrealEgo2 | 测试 RGB、姿态、SfM 和训练/验证姿态目录；测试 RGB 发现压缩分卷 | 候选双目身体数据；本轮未检查 RGB 分卷完整性或解压，不能断言训练 RGB 已有。 |
| UnrealEgo-RW | 训练/验证与测试 RGB、姿态目录，左右相机标定；测试目录有已解压 studio 数据 | 本项目已有冻结 200 张对比；本轮未验证整个数据集是否齐全。 |
| EgoFishBody | 多个动作目录、SMPL-X 参数目录 | 本地数据和拟合参数存在；影像、视角、标注来源及是否为独立真值待核实。 |
| EgoDex | 多个分卷和 unzipped 目录 | 可作为手部/交互方向候选；人体身体标注和评测适配待核实。 |

证据见 [inventory.json](evidence/inventory.json)。这里的存在表示目录/代表性文件可列出，不代表全部图片可读取、划分完整或可直接复现；部分可能位于只含压缩分卷的状态。EgoWholeBody 目前仅核实一个渲染人物目录，不能推断完整数据集都已下载。

若目标是继续比较四个检测器的 2D 泛化，可优先抽查 EgoWholeBody、SceneEgo 和 EgoPW 的图片、标注及测试划分；若目标是直接运行现有双目黄金 D，应优先检查 UnrealEgo2/RW 的成对 RGB 和相机标定。单目和外部相机数据不能直接套用当前双目拟合协议。

## 推送前自检

本轮只发布代码仓库内的简短文本和脱敏证据，不含数据、视频、图像、权重、NPY/NPZ、完整输出、凭据、.env、完整 provenance 或 patch；文件小于 50 MB。路径用数据盘/工作区占位符代替，检查工作区、暂存、待推提交和远端；历史未审计，证据见 evidence/prepush.json。
