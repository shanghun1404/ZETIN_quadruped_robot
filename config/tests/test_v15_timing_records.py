"""Retained benchmark samples must not grow the GC-traversed object graph."""
import gc
import importlib.util
from pathlib import Path

import pytest


def records():
    path=Path(__file__).resolve().parents[2]/'deploy/jetson_v15/timing_records.py'
    spec=importlib.util.spec_from_file_location('timing_records',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_partial_sample_update_preserves_presence_values_and_exact_timestamps():
    store=records().NumericRecords(8,5,{'stamp_ns':int,'received_ns':int,'compute_ms':float,'mode':str,'ok':bool})
    stamp=15;now=15114868068207
    assert stamp not in store
    store[stamp]={'stamp_ns':stamp,'ok':False}
    assert 'received_ns' not in store[stamp] and store[stamp].get('received_ns') is None
    store[stamp]['received_ns']=now
    store[stamp].update(compute_ms=11.25,mode='BLEND_STAND')
    assert dict(store[stamp])==dict(stamp_ns=stamp,received_ns=now,compute_ms=11.25,mode='BLEND_STAND',ok=False)
    assert type(store[stamp]['received_ns']) is int and type(store[stamp]['ok']) is bool
    assert list(store.items())[0][0]==stamp
    assert store.materialize()=={stamp:dict(store[stamp])}


def test_duplicate_sample_replaces_fields_and_invalid_stamp_is_rejected():
    store=records().NumericRecords(2,5,{'value':int,'other':int})
    store[0]={'value':1,'other':2};store[0]={'value':3}
    assert len(store)==1 and dict(store[0])=={'value':3}
    assert 1 not in store and -5 not in store and 10 not in store
    for stamp in (1,-5,10):
        with pytest.raises((KeyError,IndexError)):store[stamp]={'value':4}
    with pytest.raises(KeyError):store[5]['value']=9


def test_log_keeps_order_and_fails_on_overflow_instead_of_losing_evidence():
    log=records().NumericLog(2,{'stamp_ns':int,'received_ns':int})
    log.append({'stamp_ns':100,'received_ns':200})
    log.append({'stamp_ns':101,'received_ns':201})
    assert [dict(r) for r in log]==[dict(stamp_ns=100,received_ns=200),dict(stamp_ns=101,received_ns=201)]
    with pytest.raises(IndexError):log.append({'stamp_ns':102,'received_ns':202})
    assert len(log)==2
    assert log[-1]['stamp_ns']==101
    assert log[-1:]==[dict(stamp_ns=101,received_ns=201)]


def test_retained_samples_do_not_add_one_gc_container_per_record():
    store=records().NumericRecords(50000,5,{'value':int,'mode':str})
    def container_edges(value,seen=None):
        # An outer dict still traverses edges to scalar-only inner dicts even
        # after those inner dicts have been untracked by GC.
        seen=set() if seen is None else seen
        if id(value) in seen:return 0
        seen.add(id(value))
        if isinstance(value,dict):children=tuple(value.keys())+tuple(value.values())
        elif isinstance(value,(tuple,list,set)):children=value
        else:return 0
        return len(children)+sum(container_edges(child,seen) for child in children)
    initial_edges=container_edges(vars(store))
    gc.collect();before=len(gc.get_objects())
    for i in range(50000):store[i*5]={'value':i,'mode':'WALK'}
    gc.collect();after=len(gc.get_objects())
    assert len(store)==50000
    assert after-before<100, 'Timing history reintroduced a GC graph proportional to sample count'
    assert container_edges(vars(store))-initial_edges<100, 'Retained row edges still grow with sample count'
