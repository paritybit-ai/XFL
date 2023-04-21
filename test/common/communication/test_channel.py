import pytest
from unittest.mock import patch

from common.communication.gRPC.python.channel import BroadcastChannel, DualChannel, FullConnectedChannel
from common.communication.gRPC.python.commu import Commu
from service.fed_config import FedConfig

    
class TestFullConnectedChannel():
    @pytest.mark.parametrize("scheduler_id, trainer_ids, node_id", [
        ("S", ["A", "B"], "A"), ("S", ["A", "B"], "C"), ("S", ["A", "C"], "A"), ("S", ["A"], "A")
    ])
    def test_init(self, scheduler_id, trainer_ids, node_id):
        Commu.scheduler_id = scheduler_id
        Commu.trainer_ids = trainer_ids
        Commu.node_id = node_id
        
        if node_id == "C" or trainer_ids in (["A", "C"], ["A"]):
            with pytest.raises(ValueError):
                chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
        else:
            chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
            
    @pytest.mark.parametrize("accumulate_offset", [True, False])
    def test_gen_send_key(self, accumulate_offset):
        Commu.scheduler_id = "S"
        Commu.trainer_ids = ["A", "B"]
        Commu.node_id = "A"
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=accumulate_offset)
        send_key = chann._gen_send_key(remote_id='B', tag='@', accumulate_offset=accumulate_offset)
        assert send_key == "1~full~0~@~A->B"
        if accumulate_offset:
            assert chann._send_offset == 1
        else:
            assert chann._send_offset == 0
        
    @pytest.mark.parametrize("accumulate_offset", [True, False])
    def test_gen_recv_key(self, accumulate_offset):
        Commu.scheduler_id = "S"
        Commu.trainer_ids = ["A", "B"]
        Commu.node_id = "A"
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=accumulate_offset)
        send_key = chann._gen_recv_key(remote_id='B', tag='@', accumulate_offset=accumulate_offset)
        assert send_key == "1~full~0~@~B->A"
        if accumulate_offset:
            assert chann._recv_offset == 1
        else:
            assert chann._recv_offset == 0
    
    def test_send(self, mocker):
        Commu.scheduler_id = "S"
        Commu.trainer_ids = ["A", "B"]
        Commu.node_id = "A"
        mocker.patch.object(Commu, "send", return_value=0)
        
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
        res = chann._send(remote_id="B", value=123)
        assert res == 0
    
    @pytest.mark.parametrize("wait", [True, False])
    def test_recv(self, wait, mocker):
        Commu.scheduler_id = "S"
        Commu.trainer_ids = ["A", "B"]
        Commu.node_id = "A"
        
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
        mocker.patch.object(Commu, "recv", return_value=123)
        data = chann._recv(remote_id="B", wait=wait)
        assert data == 123
        assert chann._recv_offset == 1
        
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
        mocker.patch.object(Commu, "recv", return_value=None)
        data = chann._recv(remote_id="B", wait=wait)
        
        assert data is None
        if wait:
            assert chann._recv_offset == 1
        else:
            assert chann._recv_offset == 0
            
    def test_swap(self, mocker):
        Commu.scheduler_id = "S"
        Commu.trainer_ids = ["A", "B"]
        Commu.node_id = "A"
        
        chann = FullConnectedChannel(name='full', ids=["A", "B"], job_id='1', auto_offset=True)
        mocker.patch.object(chann, "_recv", return_value=123)
        mocker.patch.object(chann, "_send", return_value=0)
        data = chann._swap(remote_id='B', value=123)
        assert data == 123
        
        mocker.patch.object(chann, "_send", return_value=1)
        with pytest.raises(ValueError):
            data = chann._swap(remote_id='B', value=123)

    @pytest.mark.parametrize("parallel", [True, False])
    def test_broadcast(self, parallel, mocker):
        with patch('common.communication.gRPC.python.channel.PARALLEL', parallel):
            Commu.scheduler_id = "S"
            Commu.trainer_ids = ["A", "B", "C"]
            Commu.node_id = "A"
            
            chann = FullConnectedChannel(name='full', ids=["A", "B", "C"], job_id='1', auto_offset=True)
            mocker.patch.object(chann, "_send", return_value=0)
            res = chann._broadcast(remote_ids=["B", "C"], value=123)
            assert res == 0
            assert chann._send_offset == 1
            
            mocker.patch.object(chann, "_send", return_value=1)
            with pytest.raises(ConnectionError):
                res = chann._broadcast(remote_ids=["B", "C"], value=123)
    
    @pytest.mark.parametrize("parallel", [True, False])       
    def test_scatter(self, parallel, mocker):
        with patch('common.communication.gRPC.python.channel.PARALLEL', parallel):
            Commu.scheduler_id = "S"
            Commu.trainer_ids = ["A", "B", "C"]
            Commu.node_id = "A"
            
            chann = FullConnectedChannel(name='full', ids=["A", "B", "C"], job_id='1', auto_offset=True)
            mocker.patch.object(chann, "_send", return_value=0)
            res = chann._scatter(remote_ids=["B", "C"], values=[123, 123])
            assert res == 0
            assert chann._send_offset == 1
            
            mocker.patch.object(chann, "_send", return_value=1)
            with pytest.raises(ConnectionError):
                res = chann._scatter(remote_ids=["B", "C"], values=[123, 123])
    
    @pytest.mark.parametrize("parallel", [True, False])     
    def test_collect(sefl, parallel, mocker):
        with patch('common.communication.gRPC.python.channel.PARALLEL', parallel):
            Commu.scheduler_id = "S"
            Commu.trainer_ids = ["A", "B", "C"]
            Commu.node_id = "A"
            
            chann = FullConnectedChannel(name='full', ids=["A", "B", "C"], job_id='1', auto_offset=True)
            mocker.patch.object(chann, "_recv", return_value=123)
            data = chann._collect(remote_ids=["B", "C"])
            assert data == [123, 123]

@pytest.mark.parametrize("job_id", ['1', ''])
def test_DualChannel(job_id, mocker):
    mocker.patch.object(Commu, "send", return_value=0)
    mocker.patch.object(Commu, "recv", return_value=123)
    mocker.patch.object(Commu, "get_job_id", returen_value='1')
    
    Commu.trainer_ids = ["A", "B"]
    Commu.scheduler_id = "S"
    Commu.node_id = "A"
    
    dual_chann = DualChannel(name='dual', ids=["A", "B"], job_id=job_id, auto_offset=True)
    assert dual_chann.remote_id == "B"
    
    status = dual_chann.send(value=123, tag='@', use_pickle=True)
    assert status == 0
    
    data = dual_chann.recv(tag='@', use_pickle=True, wait=True)
    assert data == 123
    
    data = dual_chann.swap(value=123, tag='@', use_pickle=True)
    assert data == 123
   

@pytest.mark.parametrize("ids, root_id, job_id", [
    (["A", "B", "C"], "A", '1'), ([], "", "")
])
def test_BroadcastChannel(ids, root_id, job_id, mocker):
    mocker.patch.object(Commu, "send", return_value=0)
    mocker.patch.object(Commu, "recv", return_value=123)
    mocker.patch.object(Commu, "get_job_id", returen_value='1')
    mocker.patch.object(FedConfig, "get_label_trainer", return_value=["A"])
    mocker.patch.object(FedConfig, "get_trainer", return_value=["B", "C"])
    
    Commu.trainer_ids = ["A", "B", "C"]
    Commu.scheduler_id = "S"
    Commu.node_id = "A"
    
    chann = BroadcastChannel(name='dual', ids=ids, root_id=root_id, job_id=job_id, auto_offset=True)
    res = chann.broadcast(value=123)
    assert res == 0
    assert set(chann.remote_ids) == set(["B", "C"])
    
    res = chann.scatter(values=[123, 123])
    assert res == 0
    
    res = chann.collect()
    assert res == [123, 123]
    
    Commu.node_id = "B"
    chann = BroadcastChannel(name='dual', ids=ids, root_id=root_id, job_id=job_id, auto_offset=True)
    
    res = chann.send(value=123)
    assert res == 0
    
    res = chann.recv()
    assert res == 123