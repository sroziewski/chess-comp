Main changes from the first edition:

For every puzzle in both datasets we will now provide 22 success probabilitity predictions in both datasets. These are precomputed using chess engines and represent the predicted success chance of a player of given rating divided by 11 levels and two rating types (rapid and blitz). This change is meant to lower the entry bar for contestants without access to specialized hardware.

Different data and bigger datasets, this time the training dataset has over 4.5 million instances, compared to 3.7 million in the first edition.

 

In a chess puzzle, the player assumes a role of White or Black in a particular configuration of pieces on a chessboard. The goal for the puzzle taker is to find the best sequence of moves, either outright checkmating the opponent or obtaining a winning material advantage.

On the Internet, chess puzzles are often found on chess websites, like Lichess. The moves from the opposing side are made automatically and the puzzle taker is provided with immediate feedback.

Solving puzzles is considered one of the primary ways to hone chess skills. However, currently the only way to reliably estimate puzzle difficulty is to present it to a wide variety of chess players and see if they manage to solve it.

The goal of the contest is to predict how difficult a chess puzzle is from the initial position of the pieces and moves in the solution. Puzzle difficulty is measured by its Glicko-2 rating calibrated by Lichess. In simplified terms, it means that Lichess treats each attempt at solving a puzzle like a match between the user and the puzzle. If the user solves the puzzle correctly, that counts as a win for the user and they gain puzzle rating while the puzzle loses rating. When the user fails to solve the puzzle, that counts as a loss and the opposite happens. Both user and puzzle ratings are initialized at 1500.

Each chess puzzle is described by the initial position (in Forsyth–Edwards Notation, or FEN) and the moves included in the puzzle solution (in Portable Game Notation, or PGN). The solution starts with one move leading to the puzzle position and includes both moves that the puzzle taker has to find and moves by the simulated opponent.


The training and testing datasets are provided in two .csv files.

Test dataset consists of the following fields:

Field name

Field description

Field type

Example value

PuzzleId

Unique puzzle ID

string

00sHx

FEN

Standard notation for describing a particular board position of a chess game.

string

q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17

Moves

Solution to the puzzle in Portable Game Notation (PGN). Includes the last move made before the puzzle position.

string

e8d7 a2e6 d7d8 f7f8

Success_prob (22 fields)

Predicted success probabilities representing chance of correctly solving a puzzle by a player of a given rating and type of rating (rapid or blitz).

float	
0.2640107

 

Based on the above data, the challenge contestants are expected to predict the Rating field (which will be kept secret).

Field name

Field description

Field type

Example value

Rating

Puzzle Glicko-2 rating

int

1760

 

The training dataset contains all of the above fields, and also a few additional ones listed below.

RatingDeviation (int): Measure of uncertainty in the Glicko-2 rating system. It decreases as more players attempt to solve the puzzle.

Popularity (int): Users can ”upvote“ or “downvote” a puzzle. This value is the difference between the number of upvotes and downvotes.

NbPlays (int): Number of attempts at solving the puzzle.

Themes (string): Lichess allows choosing puzzles to solve based on different themes, such as tactical concepts, solution length or puzzle types (e.g. mates in x moves).

GameUrl (string): Lichess puzzles are generated from the games played on the site.

OpeningTags (string): Information about the opening from which this puzzle originated. This field has missing values.

Solution format 
Solutions in this competition should be submitted to the online evaluation system as a text file with exactly 2235 lines containing predictions for test instances. Each line in the submission should contain a single integer that indicates the predicted rating of the chess puzzle. The ordering of predictions should be the same as the ordering of the test set.

Evaluation
The quality of submissions will be evaluated using the mean squared error metric. 

Solutions will be evaluated online, and the preliminary results will be published on the public leaderboard. The public leaderboard will be available starting April 25th. The preliminary score will be computed on a subset of the test records, fixed for all participants. The final evaluation will be performed after the completion of the competition using the remaining part of the test records. Those results will also be published online. It is important to note that only teams that submit a report describing their approach before the end of the challenge will qualify for the final evaluation.


example of trainset 

PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,success_prob_rapid_1050,success_prob_rapid_1150,success_prob_rapid_1250,success_prob_rapid_1350,success_prob_rapid_1450,success_prob_rapid_1550,success_prob_rapid_1650,success_prob_rapid_1750,success_prob_rapid_1850,success_prob_rapid_1950,success_prob_rapid_2050,success_prob_blitz_1050,success_prob_blitz_1150,success_prob_blitz_1250,success_prob_blitz_1350,success_prob_blitz_1450,success_prob_blitz_1550,success_prob_blitz_1650,success_prob_blitz_1750,success_prob_blitz_1850,success_prob_blitz_1950,success_prob_blitz_2050
00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,1902,76,95,7226,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48,,0.16142186256,0.1211242373039999,0.12445170514,0.1071458031239999,0.100395522864,0.085828076226,0.0918585252199999,0.0898640247,0.093837612312,0.0978300658919999,0.0954601818239999,0.2480058670639999,0.2188178265,0.20965329144,0.186691030075,0.1644905525599999,0.1526780407039999,0.1404353612559999,0.136795666634,0.12951951654,0.119234486016,0.1177518074879999

Leaderboard
Show 
10
 entries
Search:
Rank	Team Name	Score	Submission Date
1	
ToDoFindATeamName
61763.4440	2025-05-24 11:46:03
2	
bread emoji
62917.2838	2025-06-7 17:04:29
3	
transformer_enjoyer
63289.1925	2025-06-4 18:25:00
4	
ousou
63920.0152	2025-06-6 16:15:21
5	
Cyan
88148.9919	2025-06-6 16:59:58
6	
neuro
90917.4261	2025-06-7 02:49:40
7	
dymitr
98902.7207	2025-04-28 11:54:23
8	
Mathurin
107528.4306	2025-05-22 14:39:25