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
        # base cases - check if game is over or depth reached
        if self.game.game_over(board) or depth == 0:
            if self.game.is_winner(board, self.selfPiece):
                return None, self.winscore + depth
            elif self.game.is_winner(board, self.opponentPiece):
                return None, self.losescore - depth
            elif self.game.is_full(board):
                return None, 0
            else:
                return None, self.game.heuristic_value(board, self.selfPiece)
        
        valid_moves = self.game.get_valid_moves(board)
        
        if maximizingPlayer:
            max_score = -float('inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                new_board = self.game.play_move(board, move, self.selfPiece)
                _, score = self.minimax(new_board, depth - 1, False, alpha, beta)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            
            return best_move, max_score
        
        else:
            min_score = float('inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                new_board = self.game.play_move(board, move, self.opponentPiece)
                _, score = self.minimax(new_board, depth - 1, True, alpha, beta)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                
                beta = min(beta, score)
                if beta <= alpha:
                    break
            
            return best_move, min_score