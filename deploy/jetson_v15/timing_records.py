"""Object-free sample storage for timed control benchmarks.

Allocate before streaming. Retained samples live in numeric NumPy arrays so
cyclic GC does not traverse one Python container per recorded sample.
Materialize ordinary dictionaries only after the timed stream ends.
"""
from collections.abc import MutableMapping

import numpy as np


class _Row(MutableMapping):
    def __init__(self,store,index):self.store=store;self.index=index

    def __getitem__(self,key):
        store=self.store;column=store.positions[key]
        if not int(store.present[self.index])&(1<<column):raise KeyError(key)
        value=store.columns[key][self.index].item()
        return store.strings[value] if store.fields[key] is str else value

    def __setitem__(self,key,value):
        store=self.store;column=store.positions[key]
        if store.fields[key] is str:
            if not isinstance(value,str):raise TypeError('string sample required')
            if value not in store.string_codes:
                store.string_codes[value]=len(store.strings);store.strings.append(value)
            value=store.string_codes[value]
        store.columns[key][self.index]=value
        store.present[self.index]=int(store.present[self.index])|(1<<column)

    def __delitem__(self,key):
        self[key]
        self.store.present[self.index]=int(self.store.present[self.index])&~(1<<self.store.positions[key])

    def __iter__(self):
        present=int(self.store.present[self.index])
        return (name for name,column in self.store.positions.items() if present&(1<<column))

    def __len__(self):return int(self.store.present[self.index]).bit_count()


class NumericRecords:
    def __init__(self,capacity,period_ns,fields):
        if type(capacity) is not int or capacity<1 or type(period_ns) is not int or period_ns<1:
            raise ValueError('positive integer capacity and period required')
        if not 1<=len(fields)<=64 or any(t not in (int,float,bool,str) for t in fields.values()):
            raise ValueError('1..64 numeric/string fields required')
        self.capacity=capacity;self.period_ns=period_ns;self.fields=dict(fields)
        self.positions={name:i for i,name in enumerate(fields)}
        dtypes={int:np.int64,float:np.float64,bool:np.bool_,str:np.int32}
        self.columns={name:np.zeros(capacity,dtype=dtypes[kind]) for name,kind in fields.items()}
        self.present=np.zeros(capacity,dtype=np.uint64)
        self.valid=np.zeros(capacity,dtype=np.bool_);self.count=0
        self.strings=[];self.string_codes={}

    def _index(self,stamp):
        if type(stamp) is not int or stamp<0 or stamp%self.period_ns:raise KeyError(stamp)
        index=stamp//self.period_ns
        if index>=self.capacity:raise IndexError('timing record capacity exceeded')
        return index

    def __contains__(self,stamp):
        try:return bool(self.valid[self._index(stamp)])
        except (KeyError,IndexError):return False

    def __getitem__(self,stamp):
        index=self._index(stamp)
        if not self.valid[index]:raise KeyError(stamp)
        return _Row(self,index)

    def __setitem__(self,stamp,values):
        index=self._index(stamp)
        self.present[index]=0
        row=_Row(self,index);row.update(values)
        if not self.valid[index]:self.count+=1
        self.valid[index]=True

    def get(self,stamp,default=None):return self[stamp] if stamp in self else default
    def __len__(self):return self.count

    def items(self):
        for index in range(self.capacity):
            if self.valid[index]:yield index*self.period_ns,_Row(self,index)

    def values(self):return (row for _,row in self.items())
    def materialize(self):return {stamp:dict(row) for stamp,row in self.items()}


class NumericLog:
    def __init__(self,capacity,fields):self.rows=NumericRecords(capacity,1,fields)
    def append(self,values):self.rows[len(self.rows)]=values
    def __iter__(self):return self.rows.values()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if isinstance(index,slice):
            return [dict(self.rows[i]) for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        return self.rows[index]
