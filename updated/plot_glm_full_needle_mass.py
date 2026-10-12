"""Compare GLM full-needle mass with old mass and QRScore; explicit zero imputation."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from plot_metric_correlations import average_ranks, correlation
from plot_paul_graham_mask import plt, rescore, METRICS, LABELS
from plot_qrscore_experiments import LENGTHS, DEPTHS, heatmap
from plot_three_metric_correlations import load as load_historical

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'colab-results/glm4-6v-flash-full-needle-mass-partial-export.zip'
SURVEY = 'glm4-6v-flash-no-thinking-full-needle-mass-survey'
MASK = 'glm4-6v-flash-no-thinking-paul-graham-full-needle-mass-top-mask'
OLD_ROOT = ROOT / 'datasphere-results/glm4-6v-flash-attention-mass-mask-download/glm4-6v-flash-no-thinking-paul-graham-attention-mass-mask-san-francisco'
OUT = ROOT / 'report-assets/qrscore-colab/glm-full-needle-mass'
KS = (5, 10, 15, 20, 40)
CONDITIONS = ('baseline', *(f'top{k}' for k in KS))
SCORE_NAMES = ('needle_attention_mass_v1', 'full_needle_attention_mass_v1', 'query_retrieval_score_v1')
METHODS = ('full_mass', 'old_mass', 'qr')
METHOD_LABELS = {'full_mass':'Full needle mass', 'old_mass':'Old mass', 'qr':'QRScore'}
COLORS = {'full_mass':'#268576', 'old_mass':'#3476b6', 'qr':'#b3609d'}


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def rank(x):
    return len(x)+1-average_ranks(x)


def top(x,k):
    return set(np.argsort(-x,kind='stable')[:k])


def save(fig, name, note):
    fig.supxlabel(note, fontsize=9)
    for ext in ('png','svg'):
        fig.savefig(OUT/'plots'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def load():
    with ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        survey=[json.loads(z.read(n)) for n in sorted(z.namelist())
                if n.startswith(SURVEY+'/detection/results/') and n.endswith('_results.json')]
        full=[]
        for n in sorted(z.namelist()):
            if n.startswith(MASK+'/evaluation/') and '/results/' in n and n.endswith('_results.json'):
                r=json.loads(z.read(n)); r.update(condition=r['experiment']['condition'],
                    imputed=False,source=f'{ARCHIVE.name}:{n}',prompt_hash_origin='measured')
                full.append(r)
    assert len(survey)==27 and len(full)==649
    heads=[f'{l}-{h}' for l in range(40) for h in range(32)]
    for r in survey:
        e=r['experiment']
        assert e['full_needle_attention_span']==e['qrscore_gold_span']
        assert e['retrieval_qualifying_steps'][SCORE_NAMES[0]]==e['retrieval_qualifying_steps'][SCORE_NAMES[1]]>0
        assert all(set(e['retrieval_scores'][m])==set(heads) for m in SCORE_NAMES)
    values=np.array([[mean(r['experiment']['retrieval_scores'][m][h] for r in survey)
                      for h in heads] for m in SCORE_NAMES])
    old_heads, old_values, provenance=load_historical('glm4-6v-flash')
    assert old_heads==heads
    prior_hashes={(r['case_id'],r['length'],r['depth']):r['prompt_sha256'] for r in provenance['cases']}
    assert all(prior_hashes[(r['case_id'],r['context_length'],r['depth_percent'])]==r['prompt_sha256'] for r in survey)
    spans=[]
    for r in survey:
        e=r['experiment']; a,b=e['decode_qualifying_span'],e['full_needle_attention_span']
        spans.append(dict(case_id=r['case_id'],context_length=r['context_length'],depth_percent=r['depth_percent'],
            old_start=a[0],old_end=a[1],full_start=b[0],full_end=b[1],
            old_inside_full=b[0]<=a[0]<a[1]<=b[1],qualifying_steps=e['retrieval_qualifying_steps'][SCORE_NAMES[0]]))
    rescore(full)
    key=lambda r:(r['condition'],int(r['context_length']),float(r['depth_percent']))
    by_key={key(r):r for r in full}
    assert len(by_key)==649
    baseline={(r['context_length'],r['depth_percent']):r for r in full if r['condition']=='baseline'}
    missing=[]
    for c in CONDITIONS:
        for l in LENGTHS:
            for d in DEPTHS:
                if (c,l,d) not in by_key:
                    assert c=='top40'
                    b=baseline[l,d]
                    row=dict(condition=c,context_length=l,depth_percent=d,imputed=True,
                        source='user_requested_zero_imputation',prompt_sha256=b['prompt_sha256'],
                        prompt_hash_origin='baseline_reference',generated_token_ids=[],
                        finish_reason='missing_imputed',score=0.0,**{m:0.0 for m in METRICS})
                    full.append(row); by_key[c,l,d]=row
                    missing.append(dict(condition=c,context_length=l,depth_percent=d,imputed_score=0.0))
    assert len(missing)==11 and len(full)==660
    old=[]
    for c in CONDITIONS:
        folder=OLD_ROOT/f'evaluation/context-42/evaluation/{c}'
        for path in sorted((folder/'results').glob('*_results.json')):
            r=json.loads(path.read_text()); r.update(condition=c,source=str(path.relative_to(ROOT)));old.append(r)
    rescore(old)
    with (ROOT/'report-assets/qrscore-colab/glm4-6v-flash/cases.csv').open() as f:
        qr=[r for r in csv.DictReader(f) if r['condition'] in CONDITIONS]
    lookup={'full_mass':by_key,'old_mass':{key(r):r for r in old},'qr':{key(r):r for r in qr}}
    assert all(len(x)==660 for x in lookup.values())
    analysis=[]
    for c in CONDITIONS:
        for l in LENGTHS:
            for d in DEPTHS:
                entries={method:lookup[method][c,l,d] for method in METHODS}
                assert len({r['prompt_sha256'] for r in entries.values()})==1
                f=entries['full_mass']
                row=dict(condition=c,masked_heads=0 if c=='baseline' else int(c[3:]),
                         context_length=l,depth_percent=d,imputed=f['imputed'],
                         prompt_sha256=f['prompt_sha256'],prompt_hash_origin=f['prompt_hash_origin'])
                for method,r in entries.items():
                    row[method+'_source']=r['source']
                    for metric in METRICS: row[method+'_'+metric]=float(r[metric])
                analysis.append(row)
    return heads,values,old_values,provenance,spans,analysis,missing


def main():
    heads,v,old_v,provenance,spans,rows,missing=load()
    (OUT/'plots').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#fafaf8','svg.fonttype':'none'})
    summary=[]; sensitivity=[]
    for c in CONDITIONS:
        group=[r for r in rows if r['condition']==c]
        observed=[r for r in group if not r['imputed']]
        for method in METHODS:
            summary.append(dict(condition=c,method=method,cases=110,
                imputed_cases=sum(r['imputed'] for r in group) if method=='full_mass' else 0,
                **{m:mean(r[method+'_'+m] for r in group) for m in METRICS}))
            sensitivity.append(dict(condition=c,method=method,cases=len(observed),
                **{m:mean(r[method+'_'+m] for r in observed) for m in METRICS}))
    summary_lookup={(r['condition'],r['method']):r for r in summary}
    corr={name:{'pearson':correlation(v[i],v[2]),
                'spearman':correlation(average_ranks(v[i]),average_ranks(v[2]))}
          for i,name in enumerate(('old_mass','full_mass'))}
    overlaps=[]
    changes=[]
    for k in KS:
        overlaps.append(dict(k=k,full_vs_same_run_old=len(top(v[1],k)&top(v[0],k)),
            full_vs_qr=len(top(v[1],k)&top(v[2],k)),old_vs_qr=len(top(v[0],k)&top(v[2],k)),
            full_vs_historical_mask_old=len(top(v[1],k)&top(old_v[1],k))))
        for action,indices in [('added',top(v[1],k)-top(old_v[1],k)),
                               ('removed',top(old_v[1],k)-top(v[1],k))]:
            for i in sorted(indices):
                changes.append(dict(k=k,change=action,head=heads[i],old_mass=v[0,i],full_mass=v[1,i],
                    qrscore=v[2,i],old_rank=rank(old_v[1])[i],same_run_old_rank=rank(v[0])[i],
                    full_rank=rank(v[1])[i],qr_rank=rank(v[2])[i]))
    ranks=np.array([rank(x) for x in v]); gallery=[]
    head_note='GLM · 1280 Q-heads · 27 paired cases · old/full mass: same decode steps · QRScore: prefill'
    quality_note='GLM · BF16 · no thinking · legacy_uniform decode · 110 cases/condition · hatched cells: 11 missing top40 set to 0 by request'

    fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
    layers=np.repeat(range(40),32)
    for ax,(a,b,label) in zip(axes,[(0,1,'Full needle vs old mass'),(0,2,'Old mass vs QRScore'),(1,2,'Full needle vs QRScore')]):
        points=ax.scatter(v[a],v[b],c=layers,cmap='viridis',s=14,alpha=.65,linewidths=0)
        lo=float(min(v[a].min(),v[b].min()))*.7
        ax.plot([lo,1],[lo,1],'--',color='#888')
        ax.set(xscale='log',yscale='log',xlim=(lo,1.05),ylim=(lo,1.05),
            xlabel=('Old mass','Full needle mass','QRScore')[a],ylabel=('Old mass','Full needle mass','QRScore')[b],
            title=f'{label}\nPearson {correlation(v[a],v[b]):.3f} · Spearman {correlation(average_ranks(v[a]),average_ranks(v[b])):.3f}')
    fig.colorbar(points,ax=axes,label='Layer',shrink=.8)
    save(fig,'head_scatter',head_note);gallery.append(('Score всех голов','head_scatter'))

    fig,axes=plt.subplots(1,4,figsize=(23,11),layout='constrained')
    grids=[v[0],v[1],v[1]-v[0],ranks[0]-ranks[1]]
    titles=['Old mass: expected_answer','Full needle mass','Full − old: raw score','Rank rise: old rank − full rank\npositive = full raises head']
    norm=LogNorm(vmin=float(v[:2].min()),vmax=float(v[:2].max()))
    for i,ax in enumerate(axes):
        kwargs={'norm':norm,'cmap':'viridis'} if i<2 else {
            'vmin':-max(abs(grids[i])),'vmax':max(abs(grids[i])),'cmap':'RdBu_r'}
        im=ax.imshow(grids[i].reshape(40,32),aspect='auto',**kwargs)
        ax.set(title=titles[i],xlabel='Q-head',ylabel='Layer');ax.set_xticks(range(0,32,4));ax.set_yticks(range(0,40,2))
        fig.colorbar(im,ax=ax,shrink=.75)
    save(fig,'head_maps',head_note);gallery.append(('Разница score и рангов по слоям и головам','head_maps'))

    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for i,ax in enumerate(axes[:2]):
        ax.scatter(ranks[i],ranks[2],s=12,c=layers,cmap='viridis',alpha=.6,linewidths=0)
        ax.plot([1,1280],[1,1280],'--',color='#888')
        ax.set(xlabel=('Old mass rank','Full needle rank')[i],ylabel='QRScore rank',
            title=f'Mean absolute rank error: {np.mean(abs(ranks[i]-ranks[2])):.1f}')
    axes[2].plot(KS,[r['old_vs_qr'] for r in overlaps],'o-',label='Old ∩ QRScore',color=COLORS['old_mass'])
    axes[2].plot(KS,[r['full_vs_qr'] for r in overlaps],'o-',label='Full ∩ QRScore',color=COLORS['full_mass'])
    axes[2].plot(KS,KS,'--',color='#888',label='Maximum k')
    axes[2].set(xlabel='top-k',ylabel='Shared heads',title='Top-k overlap');axes[2].legend()
    save(fig,'head_rankings',head_note);gallery.append(('Ранги и пересечения top-k','head_rankings'))

    union=sorted(top(v[0],20)|top(v[1],20)|top(v[2],20),key=lambda i:ranks[1,i])
    fig,axes=plt.subplots(2,1,figsize=(22,10),layout='constrained')
    x=np.arange(len(union))
    for i,name in enumerate(('Old mass','Full mass','QRScore')):
        axes[0].bar(x+(i-1)*.25,v[i,union],width=.25,label=name)
        axes[1].plot(x,ranks[i,union],'o-',label=name,ms=4)
    for ax in axes:
        ax.set_xticks(x,[heads[i] for i in union],rotation=90);ax.legend(ncol=3)
    axes[0].set(ylabel='Mean score',title='Union of old/full/QRScore top-20 · sorted by full-needle rank')
    axes[1].set(ylabel='Rank (1 = best)',xlabel='Layer-head');axes[1].invert_yaxis()
    save(fig,'top20_heads',head_note);gallery.append(('Конкретные головы из объединения top-20','top20_heads'))

    changed20=[r for r in changes if r['k']==20]
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    labels=[r['head']+'\n'+r['change'] for r in changed20];x=np.arange(len(changed20))
    for i,(metric,label) in enumerate([('old_mass','Old mass'),('full_mass','Full mass'),('qrscore','QRScore')]):
        axes[0].bar(x+(i-1)*.25,[r[metric] for r in changed20],width=.25,label=label)
    for metric,label in [('old_rank','Historical old mask ranking'),('full_rank','Full mass'),('qr_rank','QRScore')]:
        axes[1].plot(x,[r[metric] for r in changed20],'o-',label=label)
    for ax in axes:
        ax.set_xticks(x,labels);ax.legend()
    axes[0].set(title='The four heads changing top-20 membership',ylabel='Mean score')
    axes[1].set(title='Rank changes (1 = best)',ylabel='Rank');axes[1].invert_yaxis()
    save(fig,'top20_replacements',head_note);gallery.append(('Две замены, изменившие top20 masking','top20_replacements'))

    def matrix(group,field):
        lookup={(r['context_length'],r['depth_percent']):r[field] for r in group}
        return np.array([[lookup[l,d] for d in DEPTHS] for l in LENGTHS])

    def mark_missing(ax,group):
        for r in group:
            if r['imputed']:
                i=LENGTHS.index(r['context_length']);j=DEPTHS.index(r['depth_percent'])
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='///',edgecolor='#333',linewidth=.7))

    def overview(name,title,field,delta=False,baseline=False):
        fig,axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
        fig.suptitle(title,fontsize=18)
        conditions=CONDITIONS if not delta else CONDITIONS[1:]
        for ax,c in zip(axes.flat,conditions):
            group=[r for r in rows if r['condition']==c]
            im=heatmap(ax,matrix(group,field),c,delta=delta);mark_missing(ax,group)
        if delta:
            axes.flat[-1].axis('off')
            axes.flat[-1].text(.05,.9,'Δ < 0: lower quality with full mass\nΔ > 0: lower quality with reference\n\n/// = missing case set to zero',va='top',fontsize=12)
        fig.colorbar(im,ax=axes,shrink=.7,label='Δ quality, percentage points' if delta else 'Quality, %')
        save(fig,name,quality_note);gallery.append((title,name))

    overview('quality_rouge1','Full needle masking · ROUGE-1 recall','full_mass_rouge1_recall')
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,m,label in zip(axes,('rouge1_recall','rouge123_mean_recall','exact_match'),('ROUGE-1 recall','Mean ROUGE-1/2/3 recall','Exact phrase')):
        for method in METHODS:
            ax.plot((0,*KS),[summary_lookup[c,method][m] for c in CONDITIONS],'o-',label=METHOD_LABELS[method],color=COLORS[method])
        observed40=[r for r in sensitivity if r['condition']=='top40' and r['method']=='full_mass'][0]
        ax.scatter([40],[observed40[m]],marker='D',facecolors='none',edgecolors=COLORS['full_mass'],s=65,label='Full top40: observed 99 only')
        ax.set(title=label,xlabel='Masked heads',ylabel='Quality, %',ylim=(-2,103));ax.legend(fontsize=8)
    save(fig,'quality_comparison',quality_note);gallery.append(('Качество трёх рейтингов и чувствительность к пропускам','quality_comparison'))

    delta_summary=[]
    baseline={(r['context_length'],r['depth_percent']):r for r in rows if r['condition']=='baseline'}
    for r in rows:
        for m in METRICS:
            r['baseline_delta_'+m]=r['full_mass_'+m]-baseline[r['context_length'],r['depth_percent']]['full_mass_'+m]
            for ref in ('old_mass','qr'):
                r['delta_'+ref+'_'+m]=r['full_mass_'+m]-r[ref+'_'+m]
    overview('delta_baseline_rouge1','Δ ROUGE-1: full needle masking − its baseline','baseline_delta_rouge1_recall',delta=True)
    for ref in ('old_mass','qr'):
        for m,label in zip(METRICS,LABELS):
            overview('delta_'+ref+'_'+m,f'Δ {label}: full needle mass − {METHOD_LABELS[ref]}',
                     'delta_'+ref+'_'+m,delta=True)
        for c in CONDITIONS[1:]:
            g=[r for r in rows if r['condition']==c];obs=[r for r in g if not r['imputed']]
            delta_summary.append(dict(reference=ref,condition=c,cases=110,imputed_cases=sum(r['imputed'] for r in g),
                **{m:mean(r['delta_'+ref+'_'+m] for r in g) for m in METRICS},
                observed_cases=len(obs),observed_rouge1_delta=mean(r['delta_'+ref+'_rouge1_recall'] for r in obs)))
    head_rows=[dict(head=h,layer=i//32,q_head=i%32,old_mass=v[0,i],full_mass=v[1,i],qrscore=v[2,i],
        score_increase=v[1,i]-v[0,i],old_rank=ranks[0,i],full_rank=ranks[1,i],qr_rank=ranks[2,i]) for i,h in enumerate(heads)]
    write_csv('head_scores.csv',head_rows);write_csv('head_intersections.csv',overlaps)
    write_csv('head_topk_changes.csv',changes)
    write_csv('span_diagnostics.csv',spans);write_csv('cases_with_zero_imputation.csv',rows)
    write_csv('missing_cases_zero_filled.csv',missing);write_csv('quality_summary.csv',summary)
    write_csv('observed_subset_summary.csv',sensitivity);write_csv('delta_summary.csv',delta_summary)
    stats=dict(correlations=corr,top_k_intersections=overlaps,top_k_changes=changes,imputed_cases=11,
        observed_cases=649,analysis_cases=660,noncontained_old_spans=sum(not s['old_inside_full'] for s in spans),
        full_vs_old_pearson=correlation(v[0],v[1]),
        full_vs_old_spearman=correlation(average_ranks(v[0]),average_ranks(v[1])),
        mean_rank_error_old=float(np.mean(abs(ranks[0]-ranks[2]))),
        mean_rank_error_full=float(np.mean(abs(ranks[1]-ranks[2]))),quality_summary=summary)
    (OUT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(dict(archive=str(ARCHIVE.relative_to(ROOT)),
        sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),old_mask_source=str(OLD_ROOT.relative_to(ROOT)),
        qr_mask_source='report-assets/qrscore-colab/glm4-6v-flash/cases.csv',
        historical_survey=provenance,imputation='User requested zero for all five answer-quality metrics at 11 missing top40 positions',
        original_archive_modified=False,measured_prompt_hash_pairs=649,
        missing_prompt_hash_origin='Baseline reference only; not a measurement of the missing run',
        note='Head comparisons use the same survey run; masking comparisons use different runs with identical observed prompts.'),indent=2))
    lines=['# GLM: полный needle attention mass, обычный mass и QRScore','',
        '**11 отсутствующих top40 результатов заменены нулевыми оценками по запросу пользователя.**',
        'Это допущение анализа: нулевое качество не было измерено. Штриховка /// отмечает эти клетки.',
        'Исходный архив и manifests не изменены. Новые JSON ответов не создавались.',
        'ROUGE-1/2/3 recall, их среднее и exact phrase для пропусков = 0. Длины ответов и EOS не выдумываются.',
        'Для top40 также приводится сравнение на 99 реально сохранённых парных случаях.', '',
        '## Рейтинги голов', '',
        '27 случаев, 1280 Q-голов; old/full mass и QRScore измерены в одном survey.',
        'Old mass → expected_answer; full mass и QRScore → одна и та же полная needle.',
        'Обе decode-массы используют прежний общий token gate и одинаковое число шагов.',
        f'Старый span целиком внутри полного в {27-stats["noncontained_old_spans"]}/27 случаев.', '',
        '| Сравнение с QRScore | Pearson | Spearman |','|---|---:|---:|']
    lines += [f'| {METHOD_LABELS[n]} | {corr[n]["pearson"]:.4f} | {corr[n]["spearman"]:.4f} |' for n in ('old_mass','full_mass')]
    lines += ['', '| top-k | Full ∩ old (один survey) | Full ∩ QR | Old ∩ QR | Full ∩ прежний mass masking |','|---|---:|---:|---:|---:|']
    lines += [f'| {r["k"]} | {r["full_vs_same_run_old"]} | {r["full_vs_qr"]} | {r["old_vs_qr"]} | {r["full_vs_historical_mask_old"]} |' for r in overlaps]
    lines += ['', '## ROUGE-1 recall после masking', '',
        '| Условие | Full needle mass, с нулями | Обычный mass | QRScore |','|---|---:|---:|---:|']
    lines += [f'| {c} | '+ ' | '.join(f'{summary_lookup[c,method]["rouge1_recall"]:.2f}' for method in METHODS)+' |' for c in CONDITIONS]
    lines += ['', '## Основные наблюдения', '',
        'Расширение span немного приближает общий рейтинг к QRScore: Spearman 0.9629 → 0.9677. '
        'Старый и полный mass почти идентичны по рангу (Spearman 0.9975), но их top20 различаются.',
        'По сравнению с историческим рейтингом старого masking в top20 вошли **17-17** '
        '(в новом paired survey old rank32 → full rank15, QR rank5) и **20-9** '
        '(old rank23 → full rank20, QR rank4); вышли **24-1** и **19-3**. '
        'Именно эти две замены различают наборы голов в сравниваемых top20 masking. '
        'В новом paired survey old top20 вместо 19-3 содержит 5-7; поэтому замены '
        'между двумя рейтингами одного нового survey и фактическими masking-наборами '
        'не следует смешивать. CSV замен содержит отдельно historical old rank и same-run old rank.',
        'При top20 ROUGE-1 recall: старый mass51.29%, полный mass29.92%, QRScore29.77%. '
        'Все110 результатов этого условия измерены: нулевые замены здесь не участвуют. '
        'Этот результат согласуется с тем, что границы needle были существенным источником '
        'различий верхушки рейтинга; он не определяет вклад каждой из четырёх голов по отдельности.',
        'При top5 набор и ROUGE совпадают; при top10 полный mass немного менее разрушителен; '
        'при top15 полный mass сильнее снижает recall. Универсального превосходства нет.',
        'На 99 наблюдавшихся top40 случаях средний ROUGE-1 одинаков у полного и старого mass: '
        '24.16%. На полной сетке с нулями у полного mass21.74%, у старого23.94%; '
        'это среднее различие создаётся нулевыми заменами, а не измеренным улучшением ранжирования.', '']
    lines += ['', 'Все основные средние считаются по 110 позициям условия. Для top40 11 из них — допущенные нули.',
        'Дельта = качество full mass минус качество указанного reference, в процентных пунктах.',
        'Отрицательная дельта означает более сильное ухудшение при full-mass masking.',
        'Exact phrase — наличие нормализованного эталона подряд; это не семантическая accuracy.',
        'Корреляция scores не доказывает причинность. Masking — legacy_uniform, только decode.', '',
        '[Cases CSV](cases_with_zero_imputation.csv) · [Сводка качества](quality_summary.csv) · '
        '[99 наблюдавшихся top40 случаев](observed_subset_summary.csv) · [Дельты](delta_summary.csv)', '',
        '[Scores голов](head_scores.csv) · [Пересечения](head_intersections.csv) · [Замены top-k](head_topk_changes.csv) · '
        '[Spans](span_diagnostics.csv) · [11 нулевых замен](missing_cases_zero_filled.csv) · [Источники](provenance.json)', '']
    for title,name in gallery:
        lines += [f'## {title}','',f'![{title}](plots/{name}.png)','',f'[SVG](plots/{name}.svg)','']
    lines += ['Воспроизведение из `updated/`: `.venv/bin/python plot_glm_full_needle_mass.py`.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print(f'Created {len(gallery)} PNG + {len(gallery)} SVG figures',flush=True)


if __name__=='__main__':
    main()
