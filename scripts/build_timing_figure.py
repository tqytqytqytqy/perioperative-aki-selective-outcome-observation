from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'figures/revision_20261007';OUT.mkdir(parents=True,exist_ok=True);TMP=OUT
data=pd.read_csv(BASE/'diagnostics/timing/S28_overall_timing.csv')
daily=pd.read_csv(BASE/'diagnostics/timing/S31_elapsed_day_coverage.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':9})
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,4.25),gridspec_kw={'width_ratios':[1.4,1]})
cohorts=['INSPIRE','MOVER 2021','MOVER 2022'];colors=['#007F86','#B26A2B','#5066A1']
metrics=['Any postoperative test','Test in (0,48] h','Test in (48,168] h before recorded stop','Tests in both early and late windows']
labels=['Any test','Early window\n(0,48] h','Late window\n(48,168] h','Both windows']
y=np.arange(4)*1.25
for i,(cohort,c) in enumerate(zip(cohorts,colors)):
    g=data.loc[data.cohort.eq(cohort)].set_index('metric');v=np.array([g.loc[m,'percent'] for m in metrics])
    ax.barh(y+(i-1)*.25,v,height=.22,color=c,label=cohort)
    for j,x in enumerate(v):ax.text(x+1,y[j]+(i-1)*.25,f'{x:.1f}',fontsize=7,va='center')
    gd=daily.loc[daily.cohort.eq(cohort)]
    bx.plot(gd.elapsed_day,gd.all_eligible_pct,'o-',color=c,linewidth=1.5,markersize=3.5)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Patients tested (% of all eligible)')
ax.set_title('A  Availability and window coverage',loc='left',fontsize=10,fontweight='bold',pad=14)
bx.set_ylim(0,100);bx.set_xlim(.8,7.2);bx.set_xticks(range(1,8));bx.set_xlabel('Elapsed postoperative day');bx.set_ylabel('Patients tested (% of all eligible)')
bx.set_title('B  Elapsed-day coverage',loc='left',fontsize=10,fontweight='bold',pad=14)
for a in [ax,bx]:a.grid(axis='x' if a is ax else 'y',alpha=.15);a.set_axisbelow(True)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.105),ncol=3,frameon=False,fontsize=9)
fig.text(.02,.075,'All-eligible denominators: INSPIRE 33,394; MOVER 2021 2,802; MOVER 2022 2,587.',fontsize=7.5)
fig.text(.02,.035,'Testing is limited by recorded follow-up; these indicators do not establish adequate surveillance.',fontsize=7.5)
fig.subplots_adjust(left=.17,right=.98,top=.87,bottom=.29,wspace=.65)
fig.savefig(OUT/'Figure_1.pdf',bbox_inches='tight');fig.savefig(TMP/'Figure_1.png',dpi=300,bbox_inches='tight')
print(OUT/'Figure_1.pdf')
