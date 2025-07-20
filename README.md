# GEnetic-Multiagent-Selfplay
Official Repo for the Paper, Genetic Algorithm for Curriculum Design in Multi-Agent Reinforcement Learning (Song et al, 2024)(https://openreview.net/pdf?id=2CScZqkUPZ)


![GEMS against MAESTRO](GEMS_against_MAESTRO.gif) 

Our agent trained by GEnetic-Multiagent-Selfplay (Blue) playing against agent trained by MAESTRO (Samvelyan et al, 2023) (Red)

![GEMS against OpenAI](GEMS_against_OpenAI.gif)

Our agent trained by Genetic-Multiagent-Selfplay with 1.5 billion steps (Blue) against agent trained by OpenAI (Bansal et al, 2017) (Red)


# Contents
- Code to run the algorithm
- Weights from a model we have trained up to 1.5 billion steps interacting with the Mujoco Environment, to be used as possible future benchmark. Against the OpenAI's model, this model wins 50.9%, ties 11.9%, loses 37.2%
- This repo includes the Ant environment, which is based on OpenAI's Sumo-Ant (https://github.com/openai/multiagent-competition, Bansal et al, 2017)

# To Run
This code was tested with the following
- Python version 3.6
- OpenAI Gym version 0.9.1
- Mujoco 1.31
- mujoco-py version 0.5.7
- numpy version 1.12.1
- torch version 1.10.2


# To Run The Training
```bash
python main.py --train 1 
```

# To Play Trained Model
```bash
python main.py --train 0
```




