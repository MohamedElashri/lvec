"""Audit every recovered CMS invariant input against exact component references."""
import argparse
import csv
from decimal import localcontext
import hashlib
import importlib
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--host',choices=('m1-macos','x86_64-linux','aarch64-linux'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/tmp/lvec-cms-open-data'))
    args=parser.parse_args()
    sys.path.insert(0,str(args.collection_root/'paper/scripts'))
    from session_study import configure,host_metadata,read,write,digest,SOURCE_SHA
    from boundary_manifest import write_manifest
    args.source=configure(args)
    import awkward as ak
    import numpy as np
    import vector
    from lvec import LVec
    import lvec.utils as utils
    owned=importlib.import_module('lvec.lvec')
    from cms_data import NAME,BRANCHES,chunks,verify,load_cache,save_cache
    from cms_analysis import good_muons,calculate,check
    from ordinary_recovery import identical
    from references import invariant,root_reference,as_decimal,rounded
    from accuracy import error_metrics
    verify(args.cache/NAME)
    bank=args.cache/'eligible-four-muon.npz'
    if not bank.exists():
        parts=[]
        for columns in chunks(args.cache/NAME):
            good=good_muons(columns);count=ak.num(good)
            eligible=(count>=4)&(ak.sum(good.charge==1,axis=1)>=2)&(ak.sum(good.charge==-1,axis=1)>=2)
            parts.append({key:ak.to_packed(columns[key][eligible]) for key in BRANCHES})
        save_cache(bank,{key:ak.concatenate([part[key] for part in parts]) for key in BRANCHES})
    columns=load_cache(bank)
    assert len(columns['nMuon'])==366
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    host_metadata(args,output)
    (output/'collector.py').write_bytes(Path(__file__).read_bytes())
    (output/'collector.sha256').write_text(digest(Path(__file__))+'\n')
    dependencies=read(output/'code-inputs.csv')
    dependencies += [dict(file=name+'.py',sha256=digest(args.collection_root/'paper/scripts'/(name+'.py')))
                     for name in ('accuracy','ordinary_recovery')]
    write(output/'code-inputs.csv',dependencies)
    write(output/'inputs.csv',[dict(file=str(path),bytes=path.stat().st_size,sha256=digest(path))
                               for path in (args.cache/NAME,bank)])
    expected=calculate(columns,'lvec','rest_frame',True)
    check(expected,calculate(columns,'components','rest_frame',True))
    captured=[];depth=0
    original_invariant=utils.invariant_result
    original_owned=owned.invariant_result
    original_accurate=utils.accurate_result
    def instrument_invariant(*values,**options):
        nonlocal depth
        depth+=1
        try:return original_invariant(*values,**options)
        finally:depth-=1
    def capture(values,*arguments,**options):
        result=original_accurate(values,*arguments,**options)
        if depth:
            assert not options.get('dot') and arguments[0]=='np'
            assert options.get('where') is None
            arrays=np.broadcast_arrays(*values)
            observable='mass' if options.get('root') else 'mass2'
            for row,actual in zip(zip(*(array.flat for array in arrays)),np.asarray(result).flat):
                assert all(np.isfinite(value) for value in row) and np.isfinite(actual)
                captured.append(dict(evaluation=len(captured),observable=observable,
                    **{key+'_hex':float(value).hex() for key,value in zip(('px','py','pz','E'),row)},
                    captured_hex=float(actual).hex()))
        return result
    try:
        utils.invariant_result=owned.invariant_result=instrument_invariant
        utils.accurate_result=capture
        actual=calculate(columns,'lvec','rest_frame',True)
    finally:
        utils.invariant_result=original_invariant;owned.invariant_result=original_owned
        utils.accurate_result=original_accurate
    validations=identical(actual,expected,'cms_recovered_input_capture')
    write(output/'validation.csv',validations)
    assert captured
    write(output/'captured.csv',captured)
    unique={}
    for row in captured:
        key=(row['observable'],*(row[name+'_hex'] for name in ('px','py','pz','E')))
        if key not in unique:unique[key]=dict(row,evaluations=0)
        assert unique[key]['captured_hex']==row['captured_hex']
        unique[key]['evaluations']+=1
    results=[]
    for case,(key,row) in enumerate(unique.items()):
        observable=key[0];values=tuple(float.fromhex(x) for x in key[1:])
        squared,scale=invariant(values)
        with localcontext() as context:
            context.prec=800
            reference=as_decimal(squared) if observable=='mass2' else root_reference(squared)
            denominator=as_decimal(scale) if observable=='mass2' else as_decimal(scale).sqrt()
        p=LVec(*(np.full(17,value) for value in values))
        control=vector.array(dict(zip(('px','py','pz','E'),(np.full(17,value) for value in values))))
        x,y,z,e=map(np.float64,values)
        direct=e*e-(x*x+y*y+z*z)
        if observable=='mass':direct=np.copysign(np.sqrt(abs(direct)),direct)
        for implementation,value in (('lvec',float(getattr(p,observable)[0])),
                                     ('vector',float(getattr(control,observable)[0])),('numpy',float(direct))):
            if implementation=='lvec':assert value.hex()==row['captured_hex']
            metrics=error_metrics(value,reference,denominator)
            if implementation=='lvec':assert metrics['within_error_bound'],(case,metrics)
            results.append(dict(case=case,observable=observable,implementation=implementation,evaluations=row['evaluations'],
                **{name+'_hex':row[name+'_hex'] for name in ('px','py','pz','E')},**metrics))
    write(output/'errors.csv',results)
    (output/'summary.txt').write_text(
        f'{len(captured)} recovered CMS invariant evaluations, {len(unique)} unique observable/input pairs.\n'
        'All selected evaluations are retained without selecting by implementation error.\n'
        'Capture preserves every workflow buffer and histogram bit.\n'
        'All LVec outputs satisfy the unchanged component-scaled criterion.\n'
        'Both external controls use exactly the same Cartesian float64 components.\n'
        'Inputs come from the eligible CMS reconstruction, including its rest-frame calculations.\n'
        'This conditional numerical audit does not estimate population failure frequencies. No timing is collected.\n')
    (output/'COMPLETE').write_text(SOURCE_SHA+'\n')
    write_manifest(output)
    print((output/'summary.txt').read_text())


if __name__=='__main__':main()
