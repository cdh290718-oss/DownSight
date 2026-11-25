"""Draw the v4 documentation figure using the v3 gallery's typefaces and palette."""
from functools import lru_cache
from itertools import groupby
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT/'docs/downsight/golden-d-v4/stereo-fit.png'
im = Image.new('RGB', (1200, 940), '#FAFAF7')
d = ImageDraw.Draw(im)
faces = [subprocess.check_output(['fc-match','-f','%{file}',name],text=True)
         for name in ('DejaVu Sans','Droid Sans Fallback')]


@lru_cache(None)
def font(cjk,size):
    return ImageFont.truetype(faces[cjk],size)


def label(x,y,s,size,color,available):
    parts = [(''.join(chars),font(cjk,size)) for cjk,chars in groupby(s,lambda ch: ord(ch)>0x2e7f)]
    widths = [d.textlength(txt,font=f) for txt,f in parts]
    assert sum(widths)<=available, s
    x -= sum(widths)/2
    for (txt,f),width in zip(parts,widths):
        d.text((x,y),txt,font=f,fill=color,anchor='ls')
        x += width


def box(x1,y,x2,heading,detail,fill='#EDF3F7',edge='#C8D6DE'):
    d.rounded_rectangle((x1,y,x2,y+93),16,fill=fill,outline=edge,width=2)
    label((x1+x2)/2,y+38,heading,28,'#183C4D',x2-x1-24)
    label((x1+x2)/2,y+73,detail,20,'#324D58',x2-x1-24)


def arrow(points):
    d.line(points,fill='#54717C',width=4,joint='curve')
    x,y=points[-1]
    d.polygon([(x,y),(x-7,y-11),(x+7,y-11)],fill='#54717C')


label(600,49,'黄金 D v4 · 一对图像如何得到一具人体',34,'#183C4D',1160)
box(225,75,975,'同步双目鱼眼图像','利用双目标定，将同一具人体分别投到左右眼')
for x in (305,895):
    arrow([(600,168),(600,188),(x,188),(x,212)])
box(30,214,580,'身体观测与 v3 起点','RW20000 关节点、画内掩码与三维 Head')
box(620,214,1170,'Sapiens2 双目分割','在左右眼分别预测人体前景和部位类别')
arrow([(305,307),(305,349)])
arrow([(895,307),(895,349)])
box(30,351,580,'v2 连续拟合 → v3 候选搜索','保存 v3 最优姿态和已有的完整姿态候选')
box(620,351,1170,'建立左右眼观测边界','按置信度和边缘支持筛选，保留可用手部轮廓')
for x in (305,895):
    arrow([(x,444),(x,463),(600,463),(600,491)])
box(115,493,1085,'从 v3 姿态开始，按身体与轮廓总损失细化','72 步 Adam，每次试步恢复手指、双踝和前脚掌局部角，再检查可行性',
    '#EEF2FC','#C8D1EA')
arrow([(600,586),(600,626)])
box(115,628,1085,'比较四类完整姿态','v3 起点、最佳可行细化解、已有 v3 候选、由细化解重新搜索的候选')
arrow([(600,721),(600,761)])
box(115,763,1085,'输出总损失最低的可行人体','保存 locked 姿态与 global0，可重建左相机坐标下的三维关节和网格',
    '#E3F2EA','#ADC9BB')
label(600,907,'选姿后评估身体 PA、非 PA 与踝到趾根的方向误差',21,'#324D58',1120)
im.save(IMAGE)
