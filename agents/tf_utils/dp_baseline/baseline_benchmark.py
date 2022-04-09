# Script to generate baseline benchmark configuration using dp.cpp 
import json
from tokenize import Double
from tracemalloc import start
import numpy as np
from subprocess import Popen, PIPE

from torch import device, double

tf_utils_dir = "/home/utkarsh/Work/park/agents/tf_utils/dp_baseline"

class DevicePlDPBenchmark():
    def __init__(self, num_gpus = 5):
        self.num_gpus = num_gpus
        pass 

    def get_dp_placement(
        self, 
        node_features, 
        edge_adj_matrix,
        transfer_speed,
        model_name):
        
        json_string, file_name = self.convert_to_json_format(node_features, edge_adj_matrix, transfer_speed, model_name)
        dp_placement = self.exec_dp(json_string)
        return dp_placement

    def exec_dp(self, json_string, placement): 
        exec = f"{tf_utils_dir}/dp"
        dp = Popen([exec], stdout = PIPE, stdin = PIPE)

        byte_content = bytes(json_string, 'UTF-8')
        dp.stdin.write(byte_content)
        dp.stdin.flush()
        result = str(dp.stdout.read().strip().decode("utf-8"))

        return json.loads(result)

    def convert_to_json_format(
        self, 
        node_features, 
        edge_adj_matrix,
        transfer_speed,
        model_name):
        # Node features: {nodeId: [computationCost, size]}
        # transfer_speed is bytes transferred per second, only incurred if devices are different

        model_info = {} 
        model_info["maxSizePerFPGA"] = 10000000000000
        model_info["maxFPGAs"] = self.num_gpus
        model_info["maxCPUs"] = 0
        model_info["nodes"] = []
        model_info["edges"] = []

        count = 0
        for node in node_features: 
            node_info = {}
            node_info["name"] = f"node{count}"
            node_info["id"] = count 
            node_info["supportedOnFpga"] = 1
            node_info["cpuLatency"] = 9999999999
            node_info["fpgaLatency"] = float(node[0])
            node_info["isBackwardNode"] = False 
            
            # PARK does not care about size of node, so we set this to 1 here to ignore in the DP benchmark 
            node_info["size"] = 1
            model_info["nodes"].append(node_info)

        edge_adj_matrix = np.array(edge_adj_matrix)
        edges_src_dest = np.where(edge_adj_matrix == 1)

        count = 0
        for src, dest in zip(edges_src_dest[0], edges_src_dest[1]): 
            edge_info = {}
            edge_info["sourceId"] = int(src)
            edge_info["destId"] = int(dest)
            edge_info["cost"] = float(node_features[src][1] / transfer_speed)
            edge_info["size"] = float(node_features[src][1]) 
            model_info["edges"].append(edge_info) 

        json_string = json.dumps(model_info)
        file_name = f"{tf_utils_dir}/dp_inputs/{model_name}.json"
        with open(file_name, "w") as f:
            f.write(json_string)  

        return json_string, file_name
