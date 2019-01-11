# starcraft-reinforcement-learning-
Reinforcement learning on starcraft using pysc2.

## First level: Move to Beacon

I used vanissa policy gradient to solve this environments/

### Train the model

```
python beacon.py --training 1
```

The model will be saved every 100 epochs into the "logger" folder. The model name is "simple_save".

### Load the trained model

```
python beacon.py --model simple_save --replay 1
```

--replay 1 is used to save a replay of the game every 10 run.

### Load and train the agent

```
python beacon.py --model simple_save --training 1
```

### Watch a replay

python -m pysc2.bin.play --replay \<absolute-path\>
