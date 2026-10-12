from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import argparse
parser = argparse.ArgumentParser(description='Render the retrieval-head metrics diagram.')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'figures')
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','svg.fonttype':'none'})
P='#8247a5'; INK='#292533'; MUTED='#6a6175'; LINE='#d7cfdf'
fig=plt.figure(figsize=(18,12));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,18),ylim=(0,12));ax.axis('off')
def text(x,y,s,size=13,color=INK,bold=False,**kw):
 return ax.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',va='top',linespacing=1.5,**kw)
def box(x,y,w,h):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.0,rounding_size=0.14',facecolor='none',edgecolor=LINE,lw=1.1))
text(.55,11.65,'Retrieval head identification',28,bold=True)
text(.55,11.12,'Token copying and needle-directed attention during decoding and prefill',14,MUTED)
ax.plot([.55,17.45],[10.65,10.65],color=LINE,lw=1)
# Left panel
text(.55,10.29,'One head · attention pattern',18,bold=True)
text(.55,9.81,'Columns: source positions   /   Rows: attention queries',11,MUTED)
left=.65;bottom=3.18;cell=.71;n=8
vals=[.05,.05,.35,.40,.03,.02,.06,.04]
cmap=LinearSegmentedColormap.from_list('attention',['#171222','#ad56c7'])
for r in range(n):
 for c in range(n):
  y=bottom+(n-1-r)*cell
  face='#f4f1f7' if c<=r else 'none'
  if c in (2,3) and c<=r:face='#e9e0f1'
  if r==7:face=cmap(vals[c]*2)
  ax.add_patch(Rectangle((left+c*cell,y),cell-.02,cell-.02,facecolor=face,edgecolor=LINE,lw=.9))
  if r==7:ax.text(left+(c+.5)*cell,y+cell*.48,f'{vals[c]:.2f}',color='white',fontsize=14,ha='center',va='center')
ax.add_patch(Rectangle((left+2*cell-.035,bottom-.035),2*cell+.05,n*cell+.05,facecolor='none',edgecolor=P,lw=2))
text(left+3*cell,9.46,'needle',14,P,True,ha='center')
ax.plot([left+2*cell,left+4*cell],[9.10,9.10],color=P,lw=2)
for c,s in enumerate(['…','…','Dolores','Park','…','…','…','?']):text(left+(c+.5)*cell,9.0,s,11,P if c in (2,3) else MUTED,ha='center')
text(.65,2.91,'Generated token: “Park”',16,bold=True)
text(.65,2.51,'Highlighted columns correspond to tokens in the needle.\nThe final row shows attention at one generation step.',12,MUTED)
for j,(s,w) in enumerate([('Background',1.65),('Needle',1.65),('Question',1.65)]):
 x=.65+j*1.94;ax.add_patch(FancyBboxPatch((x,.44),w,.47,boxstyle='round,pad=0,rounding_size=.08',facecolor='#e9e0f1' if j==1 else 'none',edgecolor=P if j==1 else LINE,lw=1));ax.text(x+w/2,.675,s,ha='center',va='center',fontsize=11,color=INK)
ax.annotate('',xy=(3.25,1.0),xytext=(5.35,1.0),arrowprops={'arrowstyle':'->','color':P,'lw':1.6,'connectionstyle':'arc3,rad=.18'})
# Three equal cards; every element uses the same offsets.
x=7.05;w=10.40;h=2.78;tops=[10.30,7.27,4.24]
cards=[
 ('01','Token multiset score','A copy hit occurs when the most-attended position lies in the needle\nand contains the generated token. Repeats are capped by source counts.',r'$S_{\mathrm{token}}=\frac{\sum_v\min\left(\mathrm{hits}(v),\ \mathrm{count}_{\mathrm{needle}}(v)\right)}{\mathrm{number\ of\ needle\ tokens}}$','hits(v): valid copy hits for token ID v\ncount_needle(v): occurrences of v in the needle'),
 ('02','Decode attention mass','Sum attention over all needle tokens at each selected decoding step.\nAverage over steps whose generated token ID occurs in the needle.',r'$S_{\mathrm{mass}}=\frac{\sum_{t\in D}\sum_{i\in I_{\mathrm{needle}}} A_{t,i}}{\mathrm{number\ of\ selected\ decode\ steps}}$','D: selected decode steps   ·   I_needle: all needle token positions\nA(t, i): this head’s attention probability from decode step t to position i'),
 ('03','Query retrieval score · QRScore','For each question token, sum attention over all needle tokens during prefill.\nAverage these sums over the question tokens.',r'$S_{\mathrm{QR}}=\frac{\sum_{q\in Q}\sum_{i\in I_{\mathrm{needle}}} A_{q,i}}{\mathrm{number\ of\ question\ tokens}}$','Q: question positions   ·   I_needle: all needle token positions\nA(q, i): this head’s attention from question position q to position i')]
for top,(num,title,desc,formula,note) in zip(tops,cards):
 box(x,top-h,w,h)
 text(x+.28,top-.23,num,13,P,True)
 text(x+.80,top-.21,title,19,P,True)
 text(x+.28,top-.69,desc,12)
 text(x+.28,top-1.28,formula,21)
 text(x+.28,top-2.23,note,10.7,MUTED)
text(x,.99,'HEAD RANKING',10,P,True)
text(x+1.66,1.05,r'$\mathrm{Final\ score}=\frac{\mathrm{sum\ of\ per\!\!\!-\!\!\!example\ scores}}{\mathrm{number\ of\ examples}}$',17)
text(x,.38,'Scores are averaged separately for each metric to obtain three head rankings.',11,MUTED)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'retrieval_metrics_final.{ext}',dpi=260,transparent=True,bbox_inches='tight',pad_inches=.15)
plt.close(fig)
