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

def verify(mode):
    folder=ROOT/f'capture-{mode}';blob=(folder/'capture.bin').read_bytes();diag=json.loads((folder/'diag.json').read_text())
    assert diag['capture_done']==1 and diag['capture_rounds']==8 and diag['capture_sequence']==1
    rows=[]
    for rnd in range(8):
        samples=list(struct.iter_unpack('<II',blob[rnd*6400:(rnd+1)*6400]))
        for bus,pin in enumerate([9,6,4],1):
            row=decode_frame([(c,(v>>pin)&1) for c,v in samples]);j=rnd//2*3+bus-1
            ident,extended,data=expected(mode,j,rnd%2)
            assert (row['id'],row['extended'],row['dlc'],row['payload'])==(ident,extended,len(data),data),(mode,j,rnd,row)
            row.pop('payload')
            rows.append(dict(round=rnd,bus=bus,tx_pin=f'PB{pin}',joint=j,direction='response' if rnd%2 else 'command',**row))
    result=dict(mode=mode,passed=True,frames=len(rows),hardware_scope='MCU physical TX-pad GPIOB IDR; external loopback; no transceiver/physical CAN bus ACK',frames_decoded=rows)
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames_decoded'}))

if __name__=='__main__':
    for mode in (sys.argv[1:] or ['mit','servo']):verify(mode)
