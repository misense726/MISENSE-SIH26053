"""Reproducible map palette audit with Machado CVD simulation and CAM02-UCS distances.

Install the optional audit dependency into backend/venv/dataviz:
python -m pip install --no-deps --target backend/venv/dataviz colorspacious==1.1.2
"""
from pathlib import Path
import itertools,json,sys,re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend/venv/dataviz'))
from colorspacious import cspace_convert,deltaE
root=Path(__file__).resolve().parents[1]
source=(root/'frontend/src/visualization/palette.ts').read_text()
palette=dict(re.findall(r'\s+(\w+): "(#[0-9a-fA-F]{6})"',source.split('export const MAP_FILLS')[0]))
def rgb(value):return np.array([int(value[i:i+2],16)/255 for i in (1,3,5)])
def simulate(value,mode):
    color=rgb(value)
    if mode=='normal':return color
    return np.clip(cspace_convert(color,{'name':'sRGB1+CVD','cvd_type':mode,'severity':100},'sRGB1'),0,1)
def luminance(value):
    c=rgb(value);linear=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
    return float(linear @ [.2126,.7152,.0722])
def contrast(a,b):
    x,y=sorted([luminance(a),luminance(b)])
    return round((y+.05)/(x+.05),2)
report={'method':'colorspacious 1.1.2; Machado severity 100; CAM02-UCS delta E','background':palette['background'],'categorical':{},'text_contrast':{}}
for mode in ['normal','protanomaly','deuteranomaly','tritanomaly']:
    report['categorical'][mode]={f'{a}/{b}':round(float(deltaE(simulate(palette[a],mode),simulate(palette[b],mode))),2) for a,b in itertools.combinations(['road','obstacle','dynamic','terrain','unknown'],2)}
for color in ['ink','muted']:
    report['text_contrast'][color]=contrast(palette[color],palette['background'])
fills=dict(re.findall(r'\s+(\w+): "(#[0-9a-fA-F]{6})"',source.split('export const MAP_FILLS')[1]))
report['actual_surface_fills']={key:fills[key] for key in ['road','obstacle','dynamic','terrain','unknown']}
report['surface_all_pairs']={mode:{f'{a}/{b}':round(float(deltaE(simulate(fills[a],mode),simulate(fills[b],mode))),2) for a,b in itertools.combinations(['road','obstacle','dynamic','terrain','unknown'],2)} for mode in ['normal','protanomaly','deuteranomaly','tritanomaly']}
report['notes']=['Delta E below 10 is flagged for redundant shape or text encoding; 10 is a project screening threshold, not a WCAG rule.','Terrain and no-return cells use subdued context colors. No-return cells also have cross marks. Inspector and accessible cell list provide explicit labels.','Hazard regions use bounded rectangles, objects use 3D boxes and pedestrian cross marks.','Resolution uses a sequential ramp with exact cell-width labels.']
report['flagged_surface_pairs']={mode:[pair for pair,value in pairs.items() if value<10] for mode,pairs in report['surface_all_pairs'].items()}
report['flagged_pairs']={mode:[pair for pair,value in pairs.items() if value<10] for mode,pairs in report['categorical'].items()}
(root/'docs/palette-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert min(report['text_contrast'].values())>=4.5
assert all(pairs[key]>=10 for pairs in report['categorical'].values() for key in ['road/obstacle','road/dynamic','obstacle/dynamic'])

assert all(value>=10 for pairs in report['surface_all_pairs'].values() for value in pairs.values())
