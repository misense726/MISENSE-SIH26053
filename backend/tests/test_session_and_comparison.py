import asyncio
import random
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.schemas import ControlMessage
from backend.app.simulation.engine import SimulationEngine
from backend.app.simulation.comparison import build_comparison
from backend.app.perception.detector import OpenPCDetDSVTAdapter

@pytest.fixture
def engine():
    random.seed(31)
    np.random.seed(31)
    return SimulationEngine()

def test_scene_revision_commands_and_single_step(engine):
    engine.step_frame()
    paused=engine.handle_command(ControlMessage(action='pause'))
    assert paused['state']['is_running'] is False
    old=paused['state']
    stepped=engine.handle_command(ControlMessage(action='step'))
    assert stepped['state']['frame_id']==old['frame_id']+1
    assert stepped['state']['is_running'] is False
    changed=engine.handle_command(ControlMessage(action='set_scene',scene_id='pothole'))
    assert changed['state']['revision']==old['revision']+1
    assert engine.last_frame.scene_id=='pothole'
    bad=engine.handle_command(ControlMessage(action='set_scene',scene_id='missing'))
    assert bad['status']=='error'
    assert bad['state']==changed['state']
    assert engine.handle_command(ControlMessage(action='set_speed',speed=-1))['status']=='error'

def test_flat_road_has_zero_datum_and_is_not_a_pothole(engine):
    frame=engine.step_frame()
    observed=[cell for cell in frame.adaptive_cells if cell.point_count]
    road=[cell for cell in observed if abs(cell.x)<3 and cell.y<10]
    assert road
    assert all(abs(cell.elevation_mean)<.05 for cell in road)
    assert not any(cell.semantic_class=='pothole' for cell in observed)

def test_full_snapshot_coverage_measured_heights_and_no_live_mutation(engine):
    engine.set_scene('pothole')
    before=engine.last_frame.model_dump_json()
    snapshot=engine._latest_snapshot
    assert not snapshot.static_points.flags.writeable
    assert len(snapshot.static_points)>len(engine.last_frame.points)
    result=build_comparison(snapshot)
    assert result.frame_id==snapshot.frame_id
    assert result.scene_id==snapshot.scene_id
    assert result.revision==snapshot.revision
    assert len(result.uniform)==65536
    assert sum(row[2]**2 for row in result.uniform)==4096
    assert min(row[0]-row[2]/2 for row in result.uniform)==-32
    assert max(row[1]+row[2]/2 for row in result.uniform)==48
    no_data=[row for row in result.uniform if row[4]==0]
    assert no_data and all(row[5]=='unknown' and not row[6] for row in no_data)
    observed=next(row for row in result.uniform if row[4]>2)
    x,y,size,z,count,*_=observed
    pts=snapshot.static_points
    crop=pts[(pts[:,0]>=x-size/2)&(pts[:,0]<x+size/2)&(pts[:,1]>=y-size/2)&(pts[:,1]<y+size/2)]
    assert z==pytest.approx(float(crop[:,2].mean()),abs=.001)
    assert count==len(crop)
    assert result.metrics.uniform_memory_kb==pytest.approx(len(result.uniform)*48/1024,abs=.01)
    assert result.metrics.adaptive_memory_kb==pytest.approx(len(result.adaptive)*56/1024,abs=.01)
    assert engine.last_frame.model_dump_json()==before
    engine.step_frame()
    assert snapshot.snapshot_id!=engine._latest_snapshot.snapshot_id
    assert result.snapshot_id==snapshot.snapshot_id

def test_comparison_deduplicates_and_retains_one_result(engine):
    engine.step_frame()
    async def capture():
        a,b=await asyncio.gather(engine.capture_comparison(),engine.capture_comparison())
        assert a.snapshot_id==b.snapshot_id
        assert engine._comparison_task is None
        engine.step_frame()
        c=await engine.capture_comparison()
        assert c.snapshot_id!=a.snapshot_id
        assert engine._latest_comparison.snapshot_id==c.snapshot_id
    asyncio.run(capture())

def test_missing_snapshot_and_mapping_refresh(engine):
    with pytest.raises(ValueError):asyncio.run(engine.capture_comparison())
    engine.step_frame()
    engine.is_running=False
    timestamp=engine.last_frame.timestamp
    state=engine.handle_command(ControlMessage(action='update_weights',weights={'w1_distance':.9}))['state']
    assert not state['is_running']
    assert engine.last_frame.timestamp==timestamp
    assert engine.last_frame.revision==state['revision']
    assert engine.handle_command(ControlMessage(action='update_weights',weights={'bogus':1}))['status']=='error'

def test_dependency_availability_cannot_claim_active_model(monkeypatch):
    import sys,types
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setitem(sys.modules,'pcdet',types.SimpleNamespace())
    detector=OpenPCDetDSVTAdapter('unused-checkpoint')
    assert detector.is_fallback and detector.model is None
    assert 'Simulated' in detector.detector_source
    assert 'not implemented' in detector.status_message

def test_api_state_and_websocket_request_identity():
    with TestClient(app) as client:
        cfg=client.get('/api/config').json()
        assert cfg['session']['scene_id']
        assert cfg['memory_unit']=='KiB'
        assert cfg['memory_baseline']=='uniform_2.5d'
        assert client.get('/api/health').json()['model']['model_loaded'] is False
        with client.websocket_connect('/ws/stream') as ws:
            ws.send_json({'action':'pause','request_id':'pause-1'})
            while True:
                message=ws.receive_json()
                if message.get('request_id')=='pause-1':break
            assert not message['result']['state']['is_running']
            ws.send_json({'action':'invalid','request_id':'bad-1'})
            while True:
                message=ws.receive_json()
                if message.get('request_id')=='bad-1':break
            assert message['type']=='error'
