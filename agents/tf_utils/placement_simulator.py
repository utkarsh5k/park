


# Simulate graph placement 
from functools import total_ordering
import numpy as np 
import heapq

class QueueItem(): 
    def __init__(self, queue_entry_time, node_id, computation_cost, communication_cost):
        self.id = node_id
        self.entry_time = queue_entry_time
        self.comp_cost = computation_cost
        self.comm_cost = communication_cost
        self.exit_time = None 

    def get_total_cost(self): 
        return self.comm_cost + self.comp_cost

    # Need to make this hashable and comparable
    def __hash__(self):
        return self.node_id

    def __eq__(self, other):
        if type(other) != type(QueueItem):
            return False 

        return other.id == self.id

    def mark_processed(self, processed_time):
        self.exit_time = processed_time 

class Queue(): 
    def __init__(self, id):
        self.queue = []
        heapq.heapify(self.queue) 
        self.clock = 0
        self.id = id

        # node_id -> true/false. True is currently scheduled, false is already processed 
        self.nodes_seen = {}

    def add_to_queue(self, item): 
        # Lowest entry time = highest priority because FIFO
        if item.id not in self.nodes_seen.keys():
            heapq.heappush(self.queue, (item.entry_time, item.id, item))
            self.nodes_seen[item.id] = True

    def peek_queue(self):
        if len(self.queue) == 0: 
            return None
        return self.queue[0][2]

    def process_item(self): 
        if len(self.queue) != 0: 
            item = heapq.heappop(self.queue)[2]
            proc_cost = item.get_total_cost()
            self.clock += proc_cost
            item.mark_processed(self.clock)
            self.nodes_seen[item.id] = False
            return item

class PlacementSimulator(): 
    def __init__(self, node_features, adj_matrix, num_devices, transfer_speed):
        self.node_features = node_features
        self.adj_matrix = adj_matrix 
        self.placement = {} 
        self.simulator_clock = 0
        self.num_devices = num_devices
        self.device_queues = {}
        self.processed_nodes = {}
        self.output_cost_per_node = {}
        self.input_cost_per_node = {}
        self.transfer_speed = transfer_speed
        self.incoming_edges_per_node = {}

        for id in range(num_devices):
            self.device_queues[id] = Queue(id)

        self.preprocess()

    def preprocess(self):
        '''
        Sets the input/output communication cost for each node
        Sets the placement mapping of node_id -> device_id 
        Sets the incoming edges for each node_id, this is done because we modify the original adj_matrix for simulation
        '''
        outputs_present_on = {}
        for id, node in zip(range(len(self.node_features)), self.node_features):
            # node_features [[computationCost, outputTensorSize, placement]...]
            # Assign node to device
            self.placement[id] = node[2]
            # adj_matrix[i, j] = 1 if edge from i to j 
            outgoing_edges = [x for x in range(len(self.node_features)) if self.adj_matrix[id, x] == 1 and self.node_features[x, 2] != node[2]]
            if len(outgoing_edges) != 0: 
                self.output_cost_per_node[id] = node[1] / self.transfer_speed
            else: 
                self.output_cost_per_node[id] = 0
            
            # Get all instances of read from different devices 
            # Per input from a different device, it has to be read from shared memory 
            # Cost of read is size of input / transfer_speed
            #TODO: Once a node on a device reads output of other node on a different device
            # other nodes on this device need not read the output again 

            incoming_edges = [(x, self.node_features[x, 1] / self.transfer_speed) for x in range(len(self.node_features)) 
            if self.adj_matrix[x, id] == 1 and self.node_features[x, 2] != node[2]]
            read_cost = 0
            for incoming_edge in incoming_edges: 
                read_cost += incoming_edge[1]
            
            self.input_cost_per_node[id] = read_cost
            self.incoming_edges_per_node[id] = np.where(self.adj_matrix[:,id] == 1)[0]

    def simulate(self):
        done = False 
        while not done: 
            done = self.step()
        
        return np.max([self.device_queues[queue_id].clock for queue_id in self.device_queues.keys()])
    
    def step(self): 
        lowest_time = None
        queue_to_process = None

        # Per step, process the queue whose next item had the lowest entry time 
        for key in self.device_queues.keys(): 
            item_in_queue = self.device_queues[key].peek_queue()
            if item_in_queue is None: 
                continue
            if lowest_time is None or lowest_time > item_in_queue.entry_time: 
                lowest_time = item_in_queue.entry_time
                queue_to_process = key
        
        if queue_to_process is not None: 
            processed_item = self.device_queues[queue_to_process].process_item()
            self.remove_dependency(processed_item)

        next_nodes = self.get_next_nodes()
        if len(next_nodes) == 0:
            return True

        for node in next_nodes: 
            # If node can be processed, all its incoming edges must have been processed
            # Check which of the incoming edges source had the highest exit time, that's the entry time of the new node 
            # Cost of reading is incorporated when the item is processed and the entry time just includes 
            # cost of processing the previous node and time taken to write their output to shared memory 
            # Anything on a different device needs to wait for the output to be written to shared memory 
            # Anything on the same device can't start processing till output writing is done
            if len(self.incoming_edges_per_node[node]) == 0:
                entry_time = 0
            else:
                entry_time = np.max([self.processed_nodes[item].exit_time for item in self.incoming_edges_per_node[node]])

            queue_item = QueueItem(
                entry_time, 
                node, 
                self.node_features[node][0], 
                self.output_cost_per_node[node] + self.input_cost_per_node[node])
            
            self.device_queues[self.placement[node]].add_to_queue(queue_item)
        
        return False

    def remove_dependency(self, processed_item):
        num_nodes = len(self.adj_matrix)
        self.processed_nodes[processed_item.id] = processed_item
        #Set all outgoing nodes of processed items to 0 because this node has completed and hence 
        # irrelevant
        self.adj_matrix[processed_item.id] = [0] * num_nodes


    def get_next_nodes(self):         
        incoming_edges = self.adj_matrix.sum(axis = 0)
        # Return nodes which are ready (all inputs available) and are not already processed
        zero_indeg_nodes = np.where(incoming_edges == 0)[0]
        return [x for x in zero_indeg_nodes if x not in self.processed_nodes.keys()]

