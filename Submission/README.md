# Reinforcement Learning for Racing AI

This project implements a reinforcement learning approach using a population-based genetic algorithm to improve a racing AI model.

## How It Works

### Population-Based Training

The system uses an evolutionary approach with the following components:

1. **Population Creation**:

   - Loads the best previously trained model as the base
   - Creates 10 variations with progressively increasing mutation levels
   - The first model is kept unchanged as an "anchor" model
   - Each model is stored in the `model/population/` directory

2. **Fitness Function**:

   - Distance traveled: Heavily rewards cars that drive further (15 points per unit)
   - Speed: Rewards cars that maintain higher speeds (5 points per unit)
   - Speed consistency: Bonus for maintaining consistent speed
   - Damage penalty: Penalizes cars that take damage (20 points per damage unit)
   - Off-track penalty: Reduced penalty for going off-track (10 points per instance)

3. **Evaluation**:

   - Models run until they stagnate (no progress for 5 seconds)
   - Stagnation detection automatically switches to the next model
   - System preserves the best model ever found across all generations

4. **Evolution and Generations**:

   - The system evolves through up to 50 generations of models
   - After all models are evaluated, the population evolves:
     - Top 30% of models are kept as "elites"
     - Remaining models created through crossover and mutation of the best individuals
     - Adaptive mutation: worse-performing offspring receive higher mutation rates
   - The best model from each generation is saved

5. **Genetic Operations**:
   - **Parent Selection**: Heavily biased toward top-performing models
   - **Crossover**: Three different strategies (uniform, weighted average, layer-wise)
   - **Mutation**: Adaptive rates based on model performance

### Advanced Features

1. **Stagnation-Based Switching**:

   - Models are evaluated until they stop making progress (5 seconds without movement)
   - This prevents wasting time on non-performing models
   - No fixed evaluation time - good models can run longer

2. **Historical Best Preservation**:

   - System keeps track of the best model ever found across all generations
   - If current generation doesn't produce better models, historical best is preserved
   - This prevents performance regression between generations

3. **Smart Population Initialization**:

   - First model is always the best previous model (no mutation)
   - Subsequent models have increasing mutation levels
   - Creates a good balance between exploration and exploitation

4. **Adaptive Mutation**:
   - Lower-ranked offspring receive more mutations
   - Higher-performing models have more subtle changes
   - Mutation strength scaled down for more stable improvements

### Using the System

The reinforcement learning mode is activated during the QUALIFYING stage. During this stage:

1. The system first loads the best previously trained model as a starting point
2. Models are evaluated until they stagnate (no progress for 5 seconds)
3. After all models are evaluated, the population evolves into a new generation
4. This process continues, improving the models over time

The best model is saved automatically when:

- A population evolution occurs
- The simulation shuts down
- Maximum generations are reached

## Running the Simulation

Run the simulation in QUALIFYING mode to activate the reinforcement learning training:

```
python pyclient.py --stage 1
```

Command-line options:

- `--stage 1`: Sets the stage to QUALIFYING (required for training mode)
- `--maxEpisodes 100`: Sets the maximum number of episodes to run

After training, the best model will be available at `model/best_model.joblib`.

## Implementation Details

The fitness function balances several factors:

- Distance traveled (primary goal, 15 points per unit)
- Maximum speed achieved (secondary goal, 5 points per unit)
- Speed consistency (bonus points for maintaining speed)
- Damage avoidance (20 point penalty per damage unit)
- Staying on track (10 point penalty per off-track incident)

The genetic algorithm now uses:

- Tournament selection heavily biased toward best models
- Three different crossover strategies (uniform, weighted average, layer-wise)
- Adaptive mutation rates based on model performance
