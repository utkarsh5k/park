# Park

### What's different in this fork 

- This will contain learnt RL models over several environments 
- Will also contain attacks against said models 

### Setting up 

In the root, you will find park-stable.yml which is the conda environment needed to run the code. 
Run ```conda env create --name park-stable --file=park-stable.yml```

### Simulation interface
Similar to OpenAI Gym interface.
```
import park

env = park.make('load_balance')

obs = env.reset()
done = False

while not done:
    # act = agent.get_action(obs)
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
```

### Running stable-baselines RL algorithms on park environments 

See ```agents/cdn_caching.py``` for example. 

### Running cross-entropy attacks on environments: 

See ```attacks/cdn_attack.py``` for example. 
A cross entropy attack defines a gaussian distribution on the input. We then sample repeatedly from this gaussian and look for input sequences that 
lead in a divergence between RL algorithm's performance and the greedy baseline. When such a divergence is found, we take the input samples that successfully caused the divergence and use the mean and standard deviation of these new samples to adjust our gaussian on the input distribution. 

### Contributors

| Environment                     | env_id                            | Committers |
| -------------                   | -------------                     | ------------- |
| Adaptive video streaming        |abr, abr_sim                       | Hongzi Mao, Akshay Narayan |
| Spark cluster job scheduling    |spark, spark_sim                   | Hongzi Mao, Malte Schwarzkopf |
| SQL database query optimization |query_optimizer                    | Parimarjan Negi |
| Network congestion control      |congestion_control                 | Akshay Narayan, Frank Cangialosi |
| Network active queue management |aqm                                | Mehrdad Khani, Songtao He |
| Tensorflow device placement     |tf_placement, tf_placement_sim     | Ravichandra Addanki |
| Circuit design                  |circuit_design                     | Hanrui Wang, Jiacheng Yang |
| CDN memory caching              |cache                              | Haonan Wang, Wei-Hung Weng |
| Multi-dim database indexing     |multi_dim_index                    | Vikram Nathan |
| Account region assignment       |region_assignment                  | Ryan Marcus |
| Server load balancing           |load_balance                       | Hongzi Mao |
| Switch scheduling               |switch_scheduling                  | Ravichandra Addanki, Hongzi Mao |

### Misc
Note: to use `argparse` that is compatiable with park parameters, add parameters using
```
from park.param import parser
parser.add_argument('--new_parameter')
config = parser.parse_args()
print(config.new_parameter)
```
