# Copyright 2022 The XFL Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import threading
from typing import Any, List, Union

from common.utils.logger import logger
from service.fed_config import FedConfig
from .commu import Commu

PARALLEL = True

# Note: now only dual_channel support wait option.
# Important: if wait option is on, one should not send None object.


class FullConnectedChannel(object):
    def __init__(self, name: str, ids: list, job_id: Union[str, int] = 0, auto_offset: bool = True):
        self.name = name
        self.ids = ids
        self.job_id = str(job_id)
        
        if Commu.node_id not in ids:
            raise ValueError(f"Local node id {Commu.node_id} is not in input ids {ids}.")
        
        if len([i for i in ids if i in Commu.trainer_ids+[Commu.scheduler_id]]) != len(ids):
            raise ValueError(f"Input ids {ids} are illegal, must be in {Commu.trainer_ids+[Commu.scheduler_id]}.")
        
        if len(ids) == 1:
            raise ValueError("The created channel has only one node.")
        
        self.auto_offset = auto_offset
        self._send_offset = 0
        self._recv_offset = 0
        
    def _gen_send_key(self, remote_id: str, tag: str, accumulate_offset: bool) -> str:
        # job_id -> channel_name -> offset -> tag -> start_end_id
        send_key = '~'.join([self.job_id, self.name, str(self._send_offset), tag, Commu.node_id+'->'+remote_id])
        if self.auto_offset and accumulate_offset:
            self._send_offset += 1
        return send_key
    
    def _gen_recv_key(self, remote_id, tag: str, accumulate_offset: bool) -> str:
        # job_id -> channel_name -> offset -> tag -> start_end_id
        recv_key = '~'.join([self.job_id, self.name, str(self._recv_offset), tag, remote_id+'->'+Commu.node_id])
        if self.auto_offset and accumulate_offset:
            self._recv_offset += 1
        return recv_key
        
    def _send(self, remote_id: str, value: Any, tag: str = '@', accumulate_offset: bool = True, use_pickle: bool = True) -> int:
        key = self._gen_send_key(remote_id, tag, accumulate_offset)
        logger.debug(f"Send {key} to {remote_id}")
        status = Commu.send(key, value, remote_id, use_pickle)
        logger.debug(f"Send {key} successfully!")
        return status
    
    def _recv(self, remote_id: str, tag: str = '@', accumulate_offset: bool = True, use_pickle: bool = True, wait: bool = True) -> Any:
        key = self._gen_recv_key(remote_id, tag, accumulate_offset)
        if wait:
            logger.debug(f"Get {key}")
            
        data = Commu.recv(key, use_pickle, wait)
        
        if wait:
            logger.debug(f"Get {key} successfully!")
        else:
            if data is not None:
                logger.debug(f"Get {key}")
                logger.debug(f"Get {key} successfully!")
            else:
                if self.auto_offset and accumulate_offset:
                    self._recv_offset -= 1
           
        return data
    
    def _swap(self, remote_id: str, value: Any, tag: str = '@', use_pickle: bool = True) -> Any:
        status = self._send(remote_id, value, tag, True, use_pickle)
        if status != 0:
            raise ValueError(f"Receive response status {status} when send to remote id {remote_id}")
        
        data = self._recv(remote_id, tag, True, use_pickle)
        return data
    
    def _broadcast(self, remote_ids: List[str], value: Any, tag: str = '@', use_pickle: bool = True) -> int:
        br_status = 0
        if PARALLEL:
            thread_list = []
            result_list = [None for id in remote_ids]
            
            def func(i, *args):
                result_list[i] = self._send(*args)
                
            for i, id in enumerate(remote_ids):
                task = threading.Thread(target=func, args=(i, id, value, tag, False, use_pickle))
                thread_list.append(task)
                
            for task in thread_list:
                task.start()
            
            for task in thread_list:
                task.join()
            
            for i, status in enumerate(result_list):
                if status != 0:
                    br_status = status
                    raise ConnectionError(f"Message send to id {remote_ids[i]} not successful, response code {status}")
        else:
            for id in remote_ids:
                status = self._send(id, value, tag, False, use_pickle)
                if status != 0:
                    br_status = status
                    raise ConnectionError(f"Message send to id {id} not successful, response code {status}")
        
        self._send_offset += 1
        return br_status
    
    def _scatter(self, remote_ids: List[str], values: List[Any], tag: str = '@', use_pickle: bool = True) -> int:
        sc_status = 0
        
        if PARALLEL:
            thread_list = []
            result_list = [None for id in remote_ids]
            
            def func(i, *args):
                result_list[i] = self._send(*args)
                
            for i, id in enumerate(remote_ids):
                task = threading.Thread(target=func, args=(i, id, values[i], tag, False, use_pickle))
                thread_list.append(task)
                
            for task in thread_list:
                task.start()
            
            for task in thread_list:
                task.join()
            
            for i, status in enumerate(result_list):
                if status != 0:
                    sc_status = status
                    raise ConnectionError(f"Message send to id {remote_ids[i]} not successful, response code {status}")
        else:
            for i, id in enumerate(remote_ids):
                status = self._send(id, values[i], tag, False, use_pickle)
                if status != 0:
                    sc_status = status
                    raise ConnectionError(f"Message send to id {id} not successful, response code {status}")

        self._send_offset += 1
        return sc_status
    
    def _collect(self, remote_ids: List[str], tag: str = '@', use_pickle: bool = True) -> List[Any]:
        data = [None for i in range(len(remote_ids))]
        if PARALLEL:
            thread_list = []
            
            def func(i, *args):
                data[i] = self._recv(*args)
                
            for i, id in enumerate(remote_ids):
                task = threading.Thread(target=func, args=(i, id, tag, False, use_pickle))
                thread_list.append(task)
                
            for task in thread_list:
                task.start()
            
            for task in thread_list:
                task.join()
        else:
            for i, id in enumerate(remote_ids):
                data[i] = self._recv(id, tag, False, use_pickle)
        
        self._recv_offset += 1
        return data
    

class DualChannel(FullConnectedChannel):
    def __init__(self, name: str, ids: list, job_id: Union[str, int] = "", auto_offset: bool = True):
        """ A peer to peer channel.

        Args:
            name (str): channel name.
            ids (list): list consist of ids for two parties.
            job_id (Union[str, int], optional): job id of a federation when creating the channel,
                if it is "", job_id will be obtained from XFL framwork automatically. Defaults to "".
            auto_offset (bool, optional): whether auto accumulate the transmission times or not.
                if it is False, tag should be set manually and make sure not repeat itself for two
                communation rounds. Defaults to True.
        """
        
        if job_id == "":
            job_id = Commu.get_job_id()
        # print(job_id)
        super().__init__(name, ids, job_id=job_id, auto_offset=auto_offset)
        self.remote_id = list(set(ids) - {Commu.node_id})[0]
        
    def send(self, value: Any, tag: str = '@', use_pickle: bool = True) -> int:
        return self._send(self.remote_id, value, tag, True, use_pickle)
    
    def recv(self, tag: str = '@', use_pickle: bool = True, wait: bool = True) -> Any:
        return self._recv(self.remote_id, tag, True, use_pickle, wait)
        
    def swap(self, value: Any, tag: str = '@', use_pickle: bool = True) -> Any:
        return self._swap(self.remote_id, value, tag, use_pickle)
    

class BroadcastChannel(FullConnectedChannel):
    def __init__(self, name: str, ids: List[str] = [], root_id: str = '', job_id: Union[str, int] = "", auto_offset: bool = True):
        if not ids:
            # ids = Commu.trainer_ids
            ids = FedConfig.get_label_trainer() + FedConfig.get_trainer()

        if not root_id:
            label_trainer_list = FedConfig.get_label_trainer()
            root_id = label_trainer_list[0] if label_trainer_list else None
        if job_id == "":
            job_id = Commu.get_job_id()

        super().__init__(name, ids, job_id=job_id, auto_offset=auto_offset)
        self.root_id = root_id
        self.remote_ids = list(set(ids) - {root_id})
        
    # for root id
    def broadcast(self, value: Any, tag: str = '@', use_pickle: bool = True) -> int:
        return self._broadcast(self.remote_ids, value, tag, use_pickle)
    
    def scatter(self, values: List[Any], tag: str = '@', use_pickle: bool = True) -> int:
        return self._scatter(self.remote_ids, values, tag, use_pickle)
    
    def collect(self, tag: str = '@', use_pickle: bool = True) -> List[Any]:
        return self._collect(self.remote_ids, tag, use_pickle)
    
    # for remote ids
    def send(self, value: Any, tag: str = '@', use_pickle: bool = True) -> int:
        return self._send(self.root_id, value, tag, True, use_pickle)
    
    def recv(self, tag: str = '@', use_pickle: bool = True) -> Any:
        return self._recv(self.root_id, tag, True, use_pickle)
