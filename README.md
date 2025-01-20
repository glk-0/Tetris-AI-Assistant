# Tetris AI Assistant (uOttaHack 7 2025)

Original Tetris game code: [nuno-faria/tetris-ai](https://github.com/nuno-faria/tetris-ai.git)

A modified Tetris game that balances AI assistance with human decision-making, inspired by the NAV Challenge at uOttaHack 7. This project explores how automation and human involvement can work together effectively by providing varying levels of AI assistance to the player.

## other collaborators :
- https://github.com/RaiyanAziz55
- https://github.com/quentinheredia

## Demo


https://github.com/user-attachments/assets/969102d9-9b67-4e63-be79-c3fb938f2d9c



### Classic Tetris Mode (No AI Assistance):
The original gameplay experience.

### AI Hint Mode:
The AI suggests the next best move for the player but does not act automatically. 

### AI-Controlled Mode:
The AI places the blocks where it determines the best fit, while the player has the option to intervene.

## Requirements

- Python 3.8+
- Tensorflow/Keras
- Tensorboard
- OpenCV-Python
- Numpy
- Pillow
- Tqdm

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/RaiyanAziz55/nav-challenge.git
   cd nav-challenge
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the game with the desired mode:
   - **Classic Tetris (No Assistance):**
     ```bash
     python3 tetris_classic.py
     ```
## Commands : 
- 'e' : confirm selection
- 'w' : up(selection in menu)
- 's' : down(selection in menu)
- 'a' : move left (in game)
- 'd' : move right (in game)
- 'r' : rotate block (in game)
- 'h' : show/hide hint (in game)
- 'i' : automatically place block in AI suggested position
- 'p' : pause/resume game
- 'q' : quit game

## How It Works

### AI Assistance Levels

#### Level 0: Classic Tetris (No Assistance)
The traditional Tetris experience where the player has full control.

#### Level 1: AI Hint Mode
The AI computes and displays the best possible move for the current piece, but the final decision rests with the player. ('h' key)

#### Level 2: AI-Controlled Mode
The AI automatically places the piece in the best position, while the player can override the AI's decision if desired. ('i' key)

### AI Mechanics

#### Reinforcement Learning
The AI uses reinforcement learning to optimize gameplay. Initially, it makes random moves and learns by observing the rewards for each action. Over time, it improves its decisions by training a neural network to predict the best moves based on the current game state.

#### Training
The training process uses Q-Learning, where the agent learns to maximize future rewards. The AI evaluates all possible moves for each piece and selects the action that yields the highest predicted score. Key attributes used for training include:
- **Lines Cleared**
- **Number of Holes**
- **Bumpiness** (difference in heights between adjacent columns)
- **Total Height**

Training details:
- Replay memory size: 20,000 states
- Random sample size per training session: 512
- Discount factor: 0.95
- Neural network: 2 hidden layers with 32 neurons each, ReLU activation

## Future work : 
- Optimizing the game running time and predictions computation time for smoother gameplay.
- Adding more difficulty levels.
## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.


Special thanks to the uOttaHack 7 team and my teammates for making this project an unforgettable experience!
