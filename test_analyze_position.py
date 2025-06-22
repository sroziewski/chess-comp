import chess
import chess.engine
import sys
import os
from stockfish_features import analyze_position

def main():
    # The position from the error message
    fen = "2r3k1/2r4p/4p1p1/1p1q1pP1/p1bP1P1Q/P6R/5B2/2R3K1 b - - 5 34"
    
    # Check if Stockfish engine path is provided as command line argument
    if len(sys.argv) > 1:
        engine_path = sys.argv[1]
    else:
        # Try to find Stockfish in common locations
        common_paths = [
            "stockfish.exe",  # Current directory
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "C:\\Program Files (x86)\\Stockfish\\stockfish.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                engine_path = path
                break
        else:
            print("Error: Stockfish engine not found. Please provide the path as a command line argument.")
            print("Example: python test_analyze_position.py path/to/stockfish.exe")
            return
    
    print(f"Testing analyze_position with position: {fen}")
    print(f"Using Stockfish engine at: {engine_path}")
    
    try:
        # Call the analyze_position function with the test position
        result = analyze_position(fen, engine_path)
        
        # Print the results
        print("\nAnalysis results:")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()