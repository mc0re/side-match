class Piece:
    def __init__(self, number, patterns):
        self.number = number
        self.patterns = [int(p) for p in patterns]  # Ensure patterns are integers
    
    def rotate_clockwise(self):
        self.patterns = [self.patterns[-1]] + self.patterns[:-1]
    
    def __repr__(self):
        return f"Piece {self.number} with patterns {self.patterns}"        
