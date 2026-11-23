"""Draw the shared README diagram with explicit CJK and Latin typefaces."""
from itertools import groupby
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
im=Image.new('RGB',(1230,806),'#FAFAF7')
d=ImageDraw.Draw(im)
fonts=[subprocess.check_output(['fc-match','-f','%{file}',s]).decode() for s in ('DejaVu Sans','Droid Sans Fallback')]
def label(x,y,s,size,col):
    runs=[(''.join(cs),ImageFont.truetype(fonts[cjk],size)) for cjk,cs in groupby(s,lambda c:ord(c)>0x2e7f)]
    widths=[d.textlength(t,font=f) for t,f in runs]
    x-=sum(widths)/2
    for (t,f),w in zip(runs,widths):
        d.text((x,y+size*.37),t,font=f,fill=col,anchor='ls')
        x+=w
def box(x1,y1,x2,y2,a,b,fill='#EDF3F7',outline='#C8D6DE'):
    d.rounded_rectangle((x1,y1,x2,y2),16,fill=fill,outline=outline,width=2)
    mid=(x1+x2)//2
    label(mid,y1+29,a,26,'#183C4D')
    label(mid,y1+65,b,20,'#324D58')
def line(points):
    d.line(points,fill='#54717C',width=4,joint='curve')
    x,y=points[-1]
    d.polygon([(x,y),(x-7,y-10),(x+7,y-10)],fill='#54717C')

label(615,39,'黄金 D v3 · 从双目观测到完整人体',34,'#183C4D')
box(180,81,1050,169,'同步双目图像与标定','每眼最多 12 个身体点，Head 提供三维位置')
line([(615,169),(615,198)])
box(180,199,1050,287,'v2 连续拟合','肩髋初始化根朝向，Head 初始化平移，Adam 保留最低损失可行解')
line([(615,287),(615,315)])
box(180,316,1050,404,'生成腕部起点','在射线上试腕深度，转肩、屈肘，用原目标选解')
for x in [308,922]:
    line([(615,404),(615,424),(x,424),(x,445)])
box(48,446,568,534,'固定躯干方向','以当前根朝向和脊柱姿态为起点')
box(662,446,1182,534,'尝试其他躯干方向','根或第三段脊柱沿一个轴转 ±4°，对齐已拟合的头',fill='#EEF2FC')
for x in [308,922]:
    line([(x,534),(x,551),(615,551),(615,576)])
box(180,577,1050,665,'在每个躯干起点下联合双臂和双腿','改变腕、踝深度与肘、膝弯曲平面，合成并复核完整人体姿态')
line([(615,665),(615,693)])
box(120,694,1110,782,'按原损失选出最终可行解','同时保留 v2 原解和腕部解，输出左右眼共享的 SMPL-X 参数与三维人体',fill='#E3F2EA',outline='#ADC9BB')
im.save(ROOT/'docs/downsight/golden-d-v3/stereo-fit.png')
