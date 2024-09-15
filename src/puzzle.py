import itertools
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import os
import matplotlib.animation as animation
import numpy as np
import json

class Piece:
    def __init__(self, number, patterns):
        self.number = number
        self.patterns = [int(p) for p in patterns]  # Ensure patterns are integers
    
    def rotate_clockwise(self):
        self.patterns = [self.patterns[-1]] + self.patterns[:-1]
    
    def __repr__(self):
        return f"Piece {self.number} with patterns {self.patterns}"        

# Define the board dimensions
BOARD_SIZE = 16
NUM_PIECES = 256
NUM_PATTERNS = 23
TOTAL_MATCHES = 480
SAVE_FILE = "board_state.json"

def is_edge_position(row, col):
    return row == 0 or row == BOARD_SIZE - 1 or col == 0 or col == BOARD_SIZE - 1

def is_corner_position(row, col):
    return (row == 0 and col == 0) or (row == 0 and col == BOARD_SIZE - 1) or \
           (row == BOARD_SIZE - 1 and col == 0) or (row == BOARD_SIZE - 1 and col == BOARD_SIZE - 1)

def is_valid_placement(board, piece, row, col):
    if board[row][col] is not None:
        return False

    # Check edge and corner constraints for pieces with 0 patterns
    if piece.patterns.count(0) == 2:
        if not is_corner_position(row, col):
            return False
    elif 0 in piece.patterns:
        if not is_edge_position(row, col):
            return False
    elif is_edge_position(row, col):
        return False

    directions = [
        (-1, 0, 2, 0),  # Top: (row offset, col offset, neighbor's bottom pattern index, piece's top pattern index)
        (0, 1, 3, 1),   # Right: (row offset, col offset, neighbor's left pattern index, piece's right pattern index)
        (1, 0, 0, 2),   # Bottom: (row offset, col offset, neighbor's top pattern index, piece's bottom pattern index)
        (0, -1, 1, 3)   # Left: (row offset, col offset, neighbor's right pattern index, piece's left pattern index)
    ]

    for direction in directions:
        neighbor_row, neighbor_col, neighbor_index, piece_index = row + direction[0], col + direction[1], direction[2], direction[3]
        if 0 <= neighbor_row < BOARD_SIZE and 0 <= neighbor_col < BOARD_SIZE and board[neighbor_row][neighbor_col] is not None:
            if board[neighbor_row][neighbor_col].patterns[neighbor_index] != piece.patterns[piece_index]:
                return False
    return True

def count_matches(board, piece, row, col):
    matches = 0
    directions = [
        (-1, 0, 2, 0),  # Top
        (0, 1, 3, 1),   # Right
        (1, 0, 0, 2),   # Bottom
        (0, -1, 1, 3)   # Left
    ]

    for direction in directions:
        neighbor_row, neighbor_col, neighbor_index, piece_index = row + direction[0], col + direction[1], direction[2], direction[3]
        if 0 <= neighbor_row < BOARD_SIZE and 0 <= neighbor_col < BOARD_SIZE and board[neighbor_row][neighbor_col] is not None:
            if board[neighbor_row][neighbor_col].patterns[neighbor_index] == piece.patterns[piece_index]:
                matches += 1
    return matches

def save_board_state(board, match_counter, remaining_pieces):
    board_state = {
        "board": [[{"number": piece.number, "patterns": piece.patterns} if piece is not None else None for piece in row] for row in board],
        "match_counter": match_counter,
        "remaining_pieces": [piece.number for piece in remaining_pieces]
    }
    with open(SAVE_FILE, 'w') as f:
        json.dump(board_state, f)

def load_board_state(pieces_dict):
    if not os.path.exists(SAVE_FILE):
        return None, 0, list(pieces_dict.values()), (8, 7)

    with open(SAVE_FILE, 'r') as f:
        board_state = json.load(f)

    board = [[Piece(piece["number"], piece["patterns"]) if piece is not None else None for piece in row] for row in board_state["board"]]
    match_counter = board_state["match_counter"]
    remaining_pieces = [pieces_dict[number] for number in board_state["remaining_pieces"]]

    return board, match_counter, remaining_pieces, (8, 7)

def get_clockwise_positions(center, radius):
    row, col = center
    positions = []

    # Top side (middle to right)
    for c in range(-radius + 1, radius + 1):
        positions.append((row - radius, col + c))
    # Right side (top to bottom)
    for r in range(-radius + 1, radius + 1):
        positions.append((row + r, col + radius))
    # Bottom side (right to left)
    for c in range(radius - 1, -radius - 1, -1):
        positions.append((row + radius, col + c))
    # Left side (bottom to top)
    for r in range(radius - 1, -radius, -1):
        positions.append((row + r, col - radius))
    # Top side (left to middle)
    for c in range(-radius, 0):
        positions.append((row - radius, col + c))

    return positions


def solve_puzzle_brute_force(board, pieces, start_position):
    match_counter = 0

    def backtrack(position_index, remaining_pieces, depth=0):
        nonlocal match_counter

        if position_index >= len(positions):
            return True

        r, c = positions[position_index]
        if board[r][c] is not None:
            return backtrack(position_index + 1, remaining_pieces, depth + 1)

        for piece in remaining_pieces:
            for _ in range(4):  # Try all four rotations
                if is_valid_placement(board, piece, r, c):
                    board[r][c] = piece
                    new_remaining_pieces = remaining_pieces[:]
                    new_remaining_pieces.remove(piece)
                    matches = count_matches(board, piece, r, c)
                    match_counter += matches

                    save_board_state(board, match_counter, new_remaining_pieces)

                    # Update visualization every 100 steps to improve performance
                    if depth % 10 == 0:
                        update_board_visual(board, ax, current_pos=(r, c), match_counter=match_counter)
                        plt.pause(0.001)

                    if match_counter == TOTAL_MATCHES:
                        return True    

                    if backtrack(position_index + 1, new_remaining_pieces, depth + 1):
                        return True
                    board[r][c] = None

                    match_counter -= matches
                    board[r][c] = None    
                piece.rotate_clockwise()
        return False

    radius = 1
    positions = []
    while radius < BOARD_SIZE:
        positions.extend(get_clockwise_positions(start_position, radius))
        radius += 1

    return backtrack(0, pieces)

def update_board_visual(board, ax, current_pos=None, match_counter=0):
    ax.clear()
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_xticks(np.arange(BOARD_SIZE))
    ax.set_yticks(np.arange(BOARD_SIZE))
    ax.set_xticklabels(range(1, BOARD_SIZE + 1))
    ax.set_yticklabels(list('ABCDEFGHIJKLMNOP'))
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.yaxis.tick_left()  # Ensure y-axis labels are on the left
    ax.invert_yaxis()  # Invert the y-axis to start labels from top-left

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece is not None:
                for idx, p in enumerate(piece.patterns):
                    if idx == 0:  # Top pattern
                        triangle = [(col - 0.5, row - 0.5), (col + 0.5, row - 0.5), (col, row)]
                    elif idx == 1:  # Right pattern
                        triangle = [(col + 0.5, row - 0.5), (col + 0.5, row + 0.5), (col, row)]
                    elif idx == 2:  # Bottom pattern
                        triangle = [(col - 0.5, row + 0.5), (col + 0.5, row + 0.5), (col, row)]
                    elif idx == 3:  # Left pattern
                        triangle = [(col - 0.5, row - 0.5), (col - 0.5, row + 0.5), (col, row)]
                    
                    poly = patches.Polygon(triangle, closed=True, edgecolor='r')
                    ax.add_patch(poly)
                    ax.text(np.mean([point[0] for point in triangle]), 
                            np.mean([point[1] for point in triangle]), 
                            str(p), ha='center', va='center', color='black')

                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(col, row, str(piece.number), ha='center', va='center', color='white')

    # Highlight the current position being checked
    if current_pos:
        r, c = current_pos
        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    # Display match counter
    ax.text(0, -1, f"Matches: {match_counter}", ha='left', va='center', fontsize=12, color='black')
#    ax.set_title(f'Matches: {match_counter}')
#    plt.draw()

if __name__ == '__main__':
# Generate all pieces (Example)
    pieces_array = [
        Piece(1,[1,17,0,0]),
        Piece(2,[1,5,0,0]),
        Piece(3,[9,17,0,0]),
        Piece(4,[17,9,0,0]),
        Piece(5,[2,1,0,1]),
        Piece(6,[10,9,0,1]),
        Piece(7,[6,1,0,1]),
        Piece(8,[6,13,0,1]),
        Piece(9,[11,17,0,1]),
        Piece(10,[7,5,0,1]),
        Piece(11,[15,9,0,1]),
        Piece(12,[8,5,0,1]),
        Piece(13,[8,13,0,1]),
        Piece(14,[21,5,0,1]),
        Piece(15,[10,1,0,9]),
        Piece(16,[18,17,0,9]),
        Piece(17,[14,13,0,9]),
        Piece(18,[12,13,0,9]),
        Piece(19,[7,9,0,9]),
        Piece(20,[15,9,0,9]),
        Piece(21,[11,5,0,9]),
        Piece(22,[12,1,0,9]),
        Piece(23,[12,13,0,9]),
        Piece(24,[20,1,0,9]),
        Piece(25,[21,1,0,9]),
        Piece(26,[2,9,0,17]),   
        Piece(27,[2,17,0,17]),  
        Piece(28,[10,17,0,17]), 
        Piece(29,[18,17,0,17]), 
        Piece(30,[7,13,0,17]),  
        Piece(31,[15,9,0,17]),  
        Piece(32,[20,17,0,17]), 
        Piece(33,[8,9,0,17]),   
        Piece(34,[22,5,0,17]),  
        Piece(35,[16,13,0,17]), 
        Piece(36,[22,5,0,17]),  
        Piece(37,[18,1,0,5]),   
        Piece(38,[3,13,0,5]),   
        Piece(39,[11,13,0,5]),  
        Piece(40,[19,9,0,5]),   
        Piece(41,[19,17,0,5]),  
        Piece(42,[15,1,0,5]),   
        Piece(43,[15,9,0,5]),   
        Piece(44,[16,17,0,5]),  
        Piece(45,[4,1,0,5]),
        Piece(46,[20,5,0,5]),   
        Piece(47,[8,5,0,5]),   
        Piece(48,[16,5,0,5]),   
        Piece(49,[2,13,0,13]),  
        Piece(50,[10,1,0,17]),  
        Piece(51,[10,9,0,13]),  
        Piece(52,[6,1,0,13]),   
        Piece(53,[7,5,0,13]),   
        Piece(54,[4,5,0,13]),   
        Piece(55,[4,13,0,13]),  
        Piece(56,[8,17,0,13]),  
        Piece(57,[16,1,0,13]),  
        Piece(58,[16,13,0,13]), 
        Piece(59,[21,9,0,13]),  
        Piece(60,[22,17,0,13]), 
        Piece(61,[4,18,2,2]),   
        Piece(62,[14,7,2,2]),   
        Piece(63,[10,3,2,10]),  
        Piece(64,[2,8,2,18]),   
        Piece(65,[18,22,2,18]), 
        Piece(66,[14,14,2,18]), 
        Piece(67,[11,10,2,18]), 
        Piece(68,[20,6,2,18]),  
        Piece(69,[22,8,2,18]),  
        Piece(70,[3,7,2,3]),
        Piece(71,[7,12,2,3]),   
        Piece(72,[14,18,2,11]), 
        Piece(73,[15,4,2,11]),  
        Piece(74,[20,15,2,11]), 
        Piece(75,[8,3,2,11]),   
        Piece(76,[14,15,2,19]), 
        Piece(77,[19,15,2,19]), 
        Piece(78,[3,16,2,7]),   
        Piece(79,[20,3,2,7]),   
        Piece(80,[16,21,2,7]),  
        Piece(81,[19,18,2,15]), 
        Piece(82,[18,18,2,4]),  
        Piece(83,[11,4,2,4]),   
        Piece(84,[18,19,2,12]), 
        Piece(85,[6,14,2,12]),  
        Piece(86,[8,12,2,12]),  
        Piece(87,[16,20,2,12]), 
        Piece(88,[2,21,2,20]),  
        Piece(89,[6,22,2,20]),  
        Piece(90,[4,16,2,20]),  
        Piece(91,[11,13,2,8]),  
        Piece(92,[19,15,2,8]),  
        Piece(93,[19,4,2,8]),   
        Piece(94,[4,21,2,22]),  
        Piece(95,[12,14,2,8]),  
        Piece(96,[21,3,2,3]),   
        Piece(97,[4,19,2,22]),  
        Piece(98,[20,8,2,22]),  
        Piece(99,[21,6,2,22]),  
        Piece(100,[22,21,2,22]), 
        Piece(101,[12,15,10,10]),
        Piece(102,[12,16,10,10]),
        Piece(103,[16,19,10,10]),
        Piece(104,[22,6,10,10]), 
        Piece(105,[4,15,10,18]), 
        Piece(106,[3,8,10,6]),   
        Piece(107,[19,8,10,6]),  
        Piece(108,[4,15,10,6]),  
        Piece(109,[16,11,10,6]), 
        Piece(110,[15,12,10,14]),
        Piece(111,[12,15,10,14]),
        Piece(112,[20,19,10,3]), 
        Piece(113,[20,16,10,3]), 
        Piece(114,[14,4,10,11]), 
        Piece(115,[7,12,10,11]), 
        Piece(116,[12,11,10,11]),
        Piece(117,[22,16,10,11]),
        Piece(118,[3,21,10,19]), 
        Piece(119,[16,12,10,7]), 
        Piece(120,[8,22,10,15]), 
        Piece(121,[14,22,10,11]),
        Piece(122,[6,16,10,20]), 
        Piece(123,[14,19,10,20]),
        Piece(124,[20,15,10,20]),
        Piece(125,[12,22,10,8]), 
        Piece(126,[21,15,10,8]), 
        Piece(127,[14,6,10,16]), 
        Piece(128,[19,21,10,16]),
        Piece(129,[4,3,2,10]),   
        Piece(130,[20,8,10,16]), 
        Piece(131,[6,20,10,21]), 
        Piece(132,[12,14,10,21]),
        Piece(133,[14,16,10,22]),
        Piece(134,[11,4,10,22]), 
        Piece(135,[4,3,10,22]),  
        Piece(136,[16,20,10,22]),
        Piece(137,[20,7,18,18]), 
        Piece(138,[6,3,18,6]),   
        Piece(139,[6,11,18,6]),  
        Piece(140,[6,12,18,6]),  
        Piece(141,[19,21,18,6]), 
        Piece(142,[15,6,18,6]),  
        Piece(143,[16,12,18,6]), 
        Piece(144,[21,21,18,6]), 
        Piece(145,[3,4,19,14]),  
        Piece(146,[18,12,18,3]), 
        Piece(147,[18,22,18,3]), 
        Piece(148,[3,14,18,3]),  
        Piece(149,[15,12,18,3]), 
        Piece(150,[6,11,18,19]), 
        Piece(151,[4,22,18,19]), 
        Piece(152,[11,11,18,7]), 
        Piece(153,[11,19,18,7]), 
        Piece(154,[22,16,18,7]), 
        Piece(155,[7,7,18,7]),   
        Piece(156,[7,12,18,4]),  
        Piece(157,[22,7,18,7]),  
        Piece(158,[7,16,18,20]), 
        Piece(159,[8,6,18,20]),  
        Piece(160,[21,21,18,8]), 
        Piece(161,[6,20,18,16]), 
        Piece(162,[14,20,18,16]),
        Piece(163,[15,11,18,22]),
        Piece(164,[4,16,18,22]), 
        Piece(165,[3,4,6,14]),   
        Piece(166,[4,8,6,14]),   
        Piece(167,[3,3,6,11]),   
        Piece(168,[11,15,6,19]), 
        Piece(169,[19,21,6,19]), 
        Piece(170,[4,8,6,4]),   
        Piece(171,[20,16,6,7]),  
        Piece(172,[21,11,6,7]),  
        Piece(173,[15,15,6,15]), 
        Piece(174,[12,20,6,15]), 
        Piece(175,[7,21,6,7]),   
        Piece(176,[7,11,19,12]), 
        Piece(177,[16,11,6,20]), 
        Piece(178,[12,16,6,8]),  
        Piece(179,[8,15,6,8]),   
        Piece(180,[7,16,6,16]),  
        Piece(181,[11,16,6,21]), 
        Piece(182,[7,11,6,21]),  
        Piece(183,[19,8,14,14]), 
        Piece(184,[22,7,14,3]),  
        Piece(185,[19,12,14,11]),
        Piece(186,[8,8,14,11]),  
        Piece(187,[15,7,14,19]), 
        Piece(188,[14,21,14,7]), 
        Piece(189,[3,19,14,7]),  
        Piece(190,[16,19,14,9]), 
        Piece(191,[3,3,14,15]),  
        Piece(192,[15,20,14,15]),
        Piece(193,[11,7,14,4]),  
        Piece(194,[21,11,14,12]),
        Piece(195,[21,22,14,12]),
        Piece(196,[22,15,14,12]),
        Piece(197,[11,22,14,20]),
        Piece(198,[19,8,14,20]), 
        Piece(199,[20,20,14,20]),
        Piece(200,[19,3,14,8]),  
        Piece(201,[21,8,14,16]), 
        Piece(202,[22,7,14,16]), 
        Piece(203,[12,19,14,21]),
        Piece(204,[12,8,14,21]), 
        Piece(205,[16,3,14,21]), 
        Piece(206,[22,21,14,21]),
        Piece(207,[22,7,3,3]),   
        Piece(208,[19,22,3,11]), 
        Piece(209,[8,15,3,11]),  
        Piece(210,[11,11,3,7]),  
        Piece(211,[16,15,3,7]),  
        Piece(212,[3,16,3,15]),  
        Piece(213,[8,8,3,4]),   
        Piece(214,[3,20,3,12]),  
        Piece(215,[11,22,3,12]), 
        Piece(216,[22,21,3,12]), 
        Piece(217,[19,15,3,20]), 
        Piece(218,[4,12,3,16]),  
        Piece(219,[11,4,3,21]),  
        Piece(220,[11,16,3,22]), 
        Piece(221,[21,21,3,22]), 
        Piece(222,[21,22,3,22]), 
        Piece(223,[12,22,11,11]),
        Piece(224,[20,7,11,11]), 
        Piece(225,[16,15,11,11]),
        Piece(226,[19,15,11,7]), 
        Piece(227,[12,12,11,7]), 
        Piece(228,[19,8,11,4]),  
        Piece(229,[7,22,11,20]), 
        Piece(231,[12,20,11,8]), 
        Piece(232,[12,21,11,8]), 
        Piece(233,[19,20,19,19]),
        Piece(234,[16,4,11,7]),  
        Piece(235,[7,4,21,4]),   
        Piece(236,[7,20,19,4]),  
        Piece(237,[12,15,19,4]), 
        Piece(238,[4,16,19,12]), 
        Piece(239,[15,22,19,20]),
        Piece(240,[21,16,11,20]),
        Piece(241,[7,21,19,8]),  
        Piece(242,[11,21,19,8]), 
        Piece(243,[15,12,7,15]), 
        Piece(244,[20,8,7,15]),  
        Piece(245,[22,20,7,4]),  
        Piece(246,[16,22,7,21]), 
        Piece(247,[21,22,15,15]),
        Piece(248,[12,4,15,4]),  
        Piece(249,[4,21,15,12]), 
        Piece(250,[16,21,15,20]),
        Piece(251,[22,8,4,4]),   
        Piece(252,[8,12,4,12]),  
        Piece(253,[16,20,12,8]), 
        Piece(254,[21,16,20,16]),
        Piece(255,[16,22,20,22]),
        Piece(256,[21,22,8,22])
    ]

    pieces_dict = {piece.number: piece for piece in pieces_array}

    # Load board state
    board, match_counter, remaining_pieces, start_position = load_board_state(pieces_dict)

    # Rotate piece 139 clockwise twice before placing it on the board (if not already placed)
    if board is None:
        # Initialize the board
        board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        remaining_pieces = list(pieces_dict.values())

        # Rotate piece 139 clockwise twice before placing it on the board (if not already placed)
        piece_139 = pieces_dict[139]
        piece_139.rotate_clockwise()
        piece_139.rotate_clockwise()
        board[start_position[0]][start_position[1]] = piece_139
        remaining_pieces.remove(piece_139)
#        start_position = (8, 7)

    # Create a figure for the animation
    fig, ax = plt.subplots(1, figsize=(10, 10))
    update_board_visual(board, ax, match_counter=match_counter)

    # Attempt to solve the puzzle with the loaded state
    if solve_puzzle_brute_force(board, remaining_pieces, start_position):
        print("Puzzle solved!")
    else:
        print("No solution found.")

    plt.show()