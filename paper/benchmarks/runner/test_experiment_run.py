import argparse
import json
import numpy as np
import pytest

import experiment_run as runner
from run_management import fresh_run, open_run, workload, atomic_json, rebuild_results, Progress
from publication_benchmark import method_specs


@pytest.mark.parametrize('options,expected', [([],28),(['--cpu-budget','-1'],28),
                                             (['--cpu-budget','8'],8)])
def test_prepare_resolves_all_cpu_default_and_explicit_budget(monkeypatch,tmp_path,options,expected):
    directory=tmp_path/'all_cpus'
    monkeypatch.setattr(runner,'cpu_count',lambda:28)
    monkeypatch.setattr('publication_benchmark.os.cpu_count',lambda:28)
    monkeypatch.setattr('sys.argv',['main.py','--prepare','--output',str(directory),*options])
    runner.main()
    cfg=open_run(directory)[1]['config']
    assert cfg['cpu_budget']==expected
    assert all(m['outer']*m['inner']==expected for m in cfg['methods'])


def test_sequential_forest_tasks_stay_in_parent_process():
    import os
    from joblib import Parallel, delayed
    with runner.selection_parallelism({'outer':1}):
        pids=Parallel(n_jobs=2,prefer='threads')(delayed(os.getpid)() for _ in range(4))
    assert set(pids)=={os.getpid()}


@pytest.mark.parametrize('task',['class','reg'])
def test_backend_fix_preserves_fitted_selection(task):
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits
    rng=np.random.RandomState(9)
    X=rng.normal(size=(60,8))
    y=(X[:,0]>0).astype(int) if task=='class' else X[:,0]+rng.normal(size=60)*.1
    for method in method_specs(2,2,[.25])[1:]:
        method=dict(method,task=task)
        with threadpool_limits(limits=1), parallel_config(backend='loky',inner_max_num_threads=1):
            old=runner.fit_selection(X,y,method,.1,42,8,8)
        with threadpool_limits(limits=1), runner.selection_parallelism(method):
            new=runner.fit_selection(X,y,method,.1,42,8,8)
        assert old[1]==new[1]
        for policy in old[0]:
            np.testing.assert_array_equal(old[0][policy],new[0][policy])


def config(small=True):
    args = argparse.Namespace(smoke=small,suite='both',repeats=runner.DEFAULT_REPEATS,data_seeds=list(runner.DEFAULT_DATA_SEEDS),
                              alphas=[.1,.001,.00001],datasets=None,cpu_budget=2,outer_jobs=2,
                              shadow_fractions=[1.,.25,.1],
                              serial_shadow_fractions=list(runner.DEFAULT_SERIAL_SHADOW_FRACTIONS),
                              no_score_cache=False,
                              f1_margin=runner.DEFAULT_F1_MARGIN,r2_margin=runner.DEFAULT_R2_MARGIN,test_level=.01)
    return runner.build_config(args)


def test_full_workload_counts():
    counts=workload(config(False))
    assert counts == dict(dataset_names=11,dataset_realizations=11,cv_partitions=165,
                          selector_fits=2475,method_jobs=2640,result_rows=3465,
                          downstream_fits_upper_bound=3135,no_selection_fits=165,
                          baseline_fits_saved=330)
    names=[m['name'] for m in config(False)['methods']]
    assert names==['NoSelection','Vanilla','Greedy_jobs1_shadows1','Greedy_jobs2_shadows1',
                   'Greedy_jobs2_shadows0.25','Greedy_jobs2_shadows0.1']


@pytest.mark.parametrize('options,expected', [([], (.01,.01)),
                         (['--f1-margin','.005','--r2-margin','.02'], (.005,.02))])
def test_prepare_cli_freezes_seed_and_margin_defaults_or_overrides(monkeypatch,tmp_path,options,expected):
    directory=tmp_path/'prepared'
    monkeypatch.setattr('sys.argv',['main.py','--prepare','--cpu-budget','1','--outer-jobs','1',
                                   '--output',str(directory),*options])
    runner.main()
    cfg=open_run(directory)[1]['config']
    assert cfg['data_seeds']==[0]
    assert (cfg['f1_margin'],cfg['r2_margin'])==expected
    assert [name for name,(kind,_) in cfg['datasets'].items() if kind=='synthetic']==[
        'reference','n_features_2000','signal_0.5','kind_xor','kind_null']


def test_explicit_dataset_can_restore_an_omitted_scenario(monkeypatch,tmp_path):
    directory=tmp_path/'explicit'
    monkeypatch.setattr('sys.argv',['main.py','--prepare','--cpu-budget','1','--outer-jobs','1',
                                   '--output',str(directory),'--datasets','kind_mixed','Arcene'])
    runner.main()
    cfg=open_run(directory)[1]['config']
    assert list(cfg['datasets'])==['kind_mixed','Arcene']


def test_fresh_directory_and_protocol_checks(tmp_path):
    directory=fresh_run(config(),tmp_path/'new')
    assert open_run(directory)[1]['workload']['dataset_names']==3
    with pytest.raises(FileExistsError):
        fresh_run(config(),directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    manifest['protocol']='old-v5'
    atomic_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='old results'):
        open_run(directory)


@pytest.mark.parametrize('action',['--run','--resume','--export'])
def test_missing_manifest_has_actionable_cli_error(monkeypatch,tmp_path,capsys,action):
    directory=tmp_path/'missing_run'
    monkeypatch.setattr('sys.argv',['main.py',action,str(directory)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code==2
    stderr=capsys.readouterr().err
    assert 'No prepared run found' in stderr
    assert 'without --run' in stderr and 'Traceback' not in stderr
    assert not directory.exists()  # Never silently replace a missing run on resume.


def test_cache_preserves_scores_and_resume_skips_fits(monkeypatch,tmp_path):
    cfg=config()
    cfg['alphas']=[.1,.001,.00001]
    cfg['methods']=method_specs(2,2,[1.])
    X=np.arange(60).reshape(20,3)
    y=np.arange(20)%2
    meta=dict(Dataset='test',Data_Seed=0,Task='class',Estimator='extra_trees',
              Repeat=0,Fold=0,Split_ID='split',Partition_ID='partition',N_Train=10,N_Test=10)
    fits,scores=[],[]

    def fit(X,y,method,alpha,*args):
        fits.append((method['name'],alpha))
        mask=np.ones(3,dtype=bool)
        return ({'strict':mask,'rough_fix':mask} if method['name']=='Vanilla' else {'selected':mask}),7,3

    def score(*args):
        scores.append(1)
        return .875

    monkeypatch.setattr(runner,'fit_selection',fit)
    monkeypatch.setattr(runner,'score_mask',score)
    checkpoint={'jobs':{}}
    for key,job in runner.run_partition(X[:10],X[10:],y[:10],y[10:],{0},meta,cfg,checkpoint):
        checkpoint['jobs'][key]=job
    rows=[row for job in checkpoint['jobs'].values() for row in job['rows']]
    assert len(fits)==10 and len(scores)==1
    assert len(rows)==15 and {row['Score'] for row in rows}=={.875}
    assert sum(row['Downstream_Fit_Performed'] for row in rows)==1
    assert list(runner.run_partition(X[:10],X[10:],y[:10],y[10:],{0},meta,cfg,checkpoint))==[]
    assert len(fits)==10
    # With scoring reuse disabled, observations remain identical.
    cfg['cache_scores']=False
    uncached=list(runner.run_partition(X[:10],X[10:],y[:10],y[10:],{0},meta,cfg,{'jobs':{}}))
    assert [row['Score'] for _,job in uncached for row in job['rows']]==[row['Score'] for row in rows]
    # Atomic checkpoints recover a damaged/incomplete JSONL export.
    directory=fresh_run(cfg,tmp_path/'resume')
    atomic_json(directory/'checkpoints'/'partition.json',checkpoint)
    (directory/'fold_results.jsonl').write_text('partial-broken-json')
    target=rebuild_results(directory)
    assert len(target.read_text().splitlines())==15


def test_resume_partial_checkpoint_uses_only_current_partition_cache(monkeypatch):
    cfg=config()
    cfg['methods']=method_specs(1,1,[1.])
    X=np.ones((20,3)); y=np.arange(20)%2
    meta=dict(Dataset='test',Data_Seed=0,Task='class',Estimator='extra_trees',
              Repeat=0,Fold=0,Split_ID='split',Partition_ID='partition',N_Train=10,N_Test=10)
    counts={'fit':0,'score':0}
    def fit(X,y,method,*args):
        counts['fit']+=1
        mask=np.ones(3,dtype=bool)
        return ({'strict':mask,'rough_fix':mask} if method['name']=='Vanilla' else {'selected':mask}),7,3
    def score(*args):
        counts['score']+=1
        return .5
    monkeypatch.setattr(runner,'fit_selection',fit)
    monkeypatch.setattr(runner,'score_mask',score)
    first=next(runner.run_partition(X[:10],X[10:],y[:10],y[10:],None,meta,cfg,{'jobs':{}}))
    checkpoint={'jobs':{first[0]:first[1]}}
    remaining=list(runner.run_partition(X[:10],X[10:],y[:10],y[10:],None,meta,cfg,checkpoint))
    assert counts=={'fit':3,'score':1}
    assert len(remaining)==2


def test_progress_persists_status(tmp_path):
    with Progress(tmp_path,10,heartbeat=60) as progress:
        progress.update('test selector',completed=1)
        progress.update('stopped for test',status='paused')
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['status']=='paused' and state['completed_jobs']==1
    assert 'test selector' in (tmp_path/'run.log').read_text()


def test_execute_partition_limit_then_resume_without_repeating(monkeypatch,tmp_path):
    cfg=config()
    cfg['datasets']={'reference':cfg['datasets']['reference']}
    cfg['methods']=method_specs(1,1,[1.])
    cfg['cpu_budget']=1
    counts={'fits':0}
    def fit(X,y,method,*args):
        counts['fits']+=1
        mask=np.ones(X.shape[1],dtype=bool)
        return ({'strict':mask,'rough_fix':mask} if method['name']=='Vanilla' else {'selected':mask}),7,X.shape[1]
    monkeypatch.setattr(runner,'fit_selection',fit)
    monkeypatch.setattr(runner,'score_mask',lambda *args:.5)
    directory=fresh_run(cfg,tmp_path/'end_to_end')
    directory,manifest=open_run(directory)
    runner.execute(directory,manifest,max_partitions=1)
    assert counts['fits']==3
    assert json.loads((directory/'progress.json').read_text())['status']=='paused'
    assert (directory/'fold_metrics.csv').exists()
    assert not (directory/'noninferiority.csv').exists()  # Await all paired folds.
    runner.execute(directory,manifest)
    assert counts['fits']==12  # 4 partitions x 3 methods, including the trivial baseline.
    assert json.loads((directory/'progress.json').read_text())['status']=='complete'
    assert len((directory/'fold_results.jsonl').read_text().splitlines())==16
    import pandas as pd
    ni=pd.read_csv(directory/'noninferiority.csv')
    assert len(ni)==2 and (ni.Margin==runner.DEFAULT_F1_MARGIN).all()
    assert (ni.Baseline_Policy=='rough_fix').all()
