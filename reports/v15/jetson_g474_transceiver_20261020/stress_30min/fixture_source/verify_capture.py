"""Independent literal AK field checks against actual GPIOB TX-pad captures."""
import json
import math
from pathlib import Path
import struct
import sys
from decode_can_trace import decode_frame

ROOT=Path(__file__).resolve().parent

def lround(x): return math.floor(x+.5) if x>=0 else math.ceil(x-.5)
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def expected(mode,j,response):
    motor=j//3+1; legacy=mode=='mit' and j%3==0;q=f32(.01*(j+1))
    if mode=='mit':
        limit=12.5 if legacy else 12.56
        raw=int((q+limit)*65535/(2*limit));q=raw*2*limit/65535-limit
        data=(raw.to_bytes(2,'big')+bytes.fromhex('800000000800')) if legacy else (b'\0\0\0'+raw.to_bytes(2,'big')+bytes.fromhex('800800'))
        ident=motor if legacy else 0x800|motor;extended=not legacy
    else:
        raw=lround(q*180/math.pi*10000);data=raw.to_bytes(4,'big',signed=True)
        q=raw/10000*math.pi/180;ident=0x400|motor;extended=True
    if response:
        if legacy:
            data=bytes([motor])+raw.to_bytes(2,'big')+bytes.fromhex('8008004100')
            ident=motor;extended=False
        else:
            data=lround(q*180/math.pi*10).to_bytes(2,'big',signed=True)+bytes.fromhex('000000001900')
            ident=0x2900|motor;extended=True
    return ident,extended,data

def verify(mode,run=None):
    folder=(Path(run) if run is not None else ROOT/'pinmap-test')/f'{mode}-pad'
    blob=(folder/'capture.bin').read_bytes()
    diag=json.loads((folder/'diag.json').read_text())
    assert diag['version']==3 and diag['capture_done']==1 and diag['capture_rounds']==8
    assert diag['capture_sequence']==1 and diag['apply_calls']==0
    rows=[]
    for rnd in range(8):
        samples=list(struct.iter_unpack('<II',blob[rnd*6400:(rnd+1)*6400]))
        for bus,tx,rx in [(1,12,11),(2,29,28),(3,15,8)]:
            ident,extended,data=expected(mode,rnd//2*3+bus-1,rnd%2)
            row=dict(round=rnd,bus=bus,direction='virtual response' if rnd%2 else 'command',rx_low_samples=sum(not(p&(1<<rx)) for c,p in samples))
            for name,bit in [('tx',tx),('rx',rx)]:
                try:
                    decoded=decode_frame([(c,(p>>bit)&1) for c,p in samples])
                    decoded.pop('payload')
                    row[name]=decoded
                    row[name+'_valid']=(decoded['id'],decoded['extended'],decoded['dlc'],decoded['payload_hex'])==(ident,extended,len(data),data.hex())
                except ValueError as error:
                    row[name]=dict(error=str(error));row[name+'_valid']=False
            rows.append(row)
    result=dict(mode=mode,tx_verified=sum(x['tx_valid'] for x in rows),rx_verified=sum(x['rx_valid'] for x in rows),scope='Packed GPIOA/B physical TX/RX pins; external loopback ignores external ACK; no motor feedback')
    result['passed']=result['tx_verified']==24 and result['rx_verified']==24
    result['per_bus']={str(bus):dict(tx_verified=sum(x['tx_valid'] for x in rows if x['bus']==bus),rx_verified=sum(x['rx_valid'] for x in rows if x['bus']==bus),rx_low_samples=sum(x['rx_low_samples'] for x in rows if x['bus']==bus)) for bus in [1,2,3]}
    result['frames']=rows
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'}))
    return result['passed']

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('modes',nargs='*',choices=['mit','servo'])
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    outcomes=[verify(mode,args.run_dir) for mode in (args.modes or ['mit','servo'])]
    raise SystemExit(0 if all(outcomes) else 1)
