"""
The Agent class. Note that the agent has just a single minimax function.
The minimax function is used in a generic Game class
"""

class Agent:
    def __init__(self, game, selfPiece, opponentPiece):
        self.game = game
        self.MIN = -float('inf')
        self.MAX = float('inf')
        self.winscore =  10000000
        self.losescore = -10000000
        self.selfPiece = selfPiece          #AI
        self.opponentPiece = opponentPiece  #HUMAN


    '''
    This is the function you must complete
    Inputs -
    1. self (Agent object reference)
    2. board (2d array)
    3. depth (float)
    4. maximizingPlayer (bool)
    5. alpha (float)
    6. beta (float)

    Outputs -
    1. next_move (type discussed below)
    2. score (float)

    Notes
    1. Need to create general implementation that considers all move types - tuple or float
    2. We use depth here to measure how far down we need to keep going, we stop at depth = 0
    3. Need to consider the different situations when game is over
    4. Prioritize winning in fewest moves
    5. next_move is the best possible move from valid_moves based on alpha-beta pruning. 
       Do not need to focus on its type, since it differs based on game, just find best move from valid_moves
    '''
    def minimax(self, board, depth, maximizingPlayer, alpha, beta):
        valid_moves = self.game.get_valid_moves(board)
        next_move, score = None, None
        return next_move, score
