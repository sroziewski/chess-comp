"""
Module for predicting and handling chess opening tags.

This module implements an advanced approach to opening tag prediction with:
1. Ensemble learning: Combines multiple models (RandomForest, GradientBoosting, SVM)
   for more robust predictions
2. Hierarchical classification: First predicts the opening family, then the variation
   within that family
3. ECO code integration: Uses ECO (Encyclopedia of Chess Openings) codes to enhance
   prediction accuracy and confidence
"""

import concurrent.futures
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes
from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.utils.config import get_config
from chess_puzzle_rating.utils.progress import get_logger

# Comprehensive list of chess opening tags
OPENING_TAGS = [
    "Alekhine_Defense",
    "Alekhine_Defense_Balogh_Variation",
    "Alekhine_Defense_Brooklyn_Variation",
    "Alekhine_Defense_Exchange_Variation",
    "Alekhine_Defense_Four_Pawns_Attack",
    "Alekhine_Defense_Hunt_Variation",
    "Alekhine_Defense_Kmoch_Variation",
    "Alekhine_Defense_Krejcik_Variation",
    "Alekhine_Defense_Maroczy_Variation",
    "Alekhine_Defense_Modern_Variation",
    "Alekhine_Defense_Mokele_Mbembe",
    "Alekhine_Defense_Normal_Variation",
    "Alekhine_Defense_OSullivan_Gambit",
    "Alekhine_Defense_Other_variations",
    "Alekhine_Defense_Samisch_Attack",
    "Alekhine_Defense_Scandinavian_Variation",
    "Alekhine_Defense_Spielmann_Gambit",
    "Alekhine_Defense_Steiner_Variation",
    "Alekhine_Defense_The_Squirrel",
    "Alekhine_Defense_Two_Pawn_Attack",
    "Alekhine_Defense_Two_Pawns_Attack",
    "Alekhine_Defense_Welling_Variation",
    "Amar_Gambit",
    "Amar_Gambit_Other_variations",
    "Amar_Opening",
    "Amar_Opening_Other_variations",
    "Amar_Opening_Paris_Gambit",
    "Amazon_Attack",
    "Amazon_Attack_Other_variations",
    "Amazon_Attack_Siberian_Attack",
    "Amsterdam_Attack",
    "Amsterdam_Attack_Other_variations",
    "Anderssens_Opening",
    "Anderssens_Opening_Other_variations",
    "Anderssens_Opening_Polish_Gambit",
    "Australian_Defense",
    "Australian_Defense_Other_variations",
    "Barnes_Defense",
    "Barnes_Defense_Other_variations",
    "Barnes_Opening",
    "Barnes_Opening_Hammerschlag",
    "Barnes_Opening_Other_variations",
    "Barnes_Opening_Walkerling",
    "Benko_Gambit",
    "Benko_Gambit_Accepted",
    "Benko_Gambit_Accepted_Central_Storming_Variation",
    "Benko_Gambit_Accepted_Dlugy_Variation",
    "Benko_Gambit_Accepted_Fianchetto_Variation",
    "Benko_Gambit_Accepted_Fully_Accepted_Variation",
    "Benko_Gambit_Accepted_King_Walk_Variation",
    "Benko_Gambit_Accepted_Modern_Variation",
    "Benko_Gambit_Accepted_Other_variations",
    "Benko_Gambit_Accepted_Pawn_Return_Variation",
    "Benko_Gambit_Accepted_Yugoslav",
    "Benko_Gambit_Declined",
    "Benko_Gambit_Declined_Bishop_Attack",
    "Benko_Gambit_Declined_Hjrring_Countergambit",
    "Benko_Gambit_Declined_Main_Line",
    "Benko_Gambit_Declined_Pseudo-Samisch",
    "Benko_Gambit_Declined_Quiet_Line",
    "Benko_Gambit_Declined_Sosonko_Variation",
    "Benko_Gambit_Fianchetto_Variation",
    "Benko_Gambit_Nescafe_Frappe_Attack",
    "Benko_Gambit_Other_variations",
    "Benko_Gambit_Zaitsev_System",
    "Benko_Gambit_Zaitsev_Variation",
    "Benoni_Defense",
    "Benoni_Defense_Benoni-Indian_Defense",
    "Benoni_Defense_Benoni-Staunton_Gambit",
    "Benoni_Defense_Benoni_Gambit",
    "Benoni_Defense_Benoni_Gambit_Accepted",
    "Benoni_Defense_Classical",
    "Benoni_Defense_Classical_Variation",
    "Benoni_Defense_Cormorant_Gambit",
    "Benoni_Defense_Czech_Benoni_Defense",
    "Benoni_Defense_Fianchetto_Variation",
    "Benoni_Defense_Four_Pawns_Attack",
    "Benoni_Defense_Franco-Sicilian_Hybrid",
    "Benoni_Defense_French_Benoni",
    "Benoni_Defense_Hawk_Variation",
    "Benoni_Defense_Hromadka_System",
    "Benoni_Defense_Kings_Indian_System",
    "Benoni_Defense_Kings_Pawn_Line",
    "Benoni_Defense_Knights_Tour_Variation",
    "Benoni_Defense_Mikenas_Variation",
    "Benoni_Defense_Modern_Variation",
    "Benoni_Defense_Old_Benoni",
    "Benoni_Defense_Other_variations",
    "Benoni_Defense_Pawn_Storm_Variation",
    "Benoni_Defense_Semi-Benoni",
    "Benoni_Defense_Snail_Variation",
    "Benoni_Defense_Taimanov_Variation",
    "Benoni_Defense_Uhlmann_Variation",
    "Benoni_Defense_Weenink_Variation",
    "Benoni_Defense_Woozle",
    "Benoni_Defense_Zilbermints-Benoni_Gambit",
    "Bird_Opening",
    "Bird_Opening_Batavo-Polish_Attack",
    "Bird_Opening_Double_Duck_Formation",
    "Bird_Opening_Dutch_Variation",
    "Bird_Opening_Froms_Gambit",
    "Bird_Opening_Hobbs-Zilbermints_Gambit",
    "Bird_Opening_Hobbs_Gambit",
    "Bird_Opening_Horsefly_Defense",
    "Bird_Opening_Lasker_Gambit",
    "Bird_Opening_Lasker_Variation",
    "Bird_Opening_Mujannah",
    "Bird_Opening_Myers_Defense",
    "Bird_Opening_Other_variations",
    "Bird_Opening_Schlechter_Gambit",
    "Bird_Opening_Sturm_Gambit",
    "Bird_Opening_Thomas_Gambit",
    "Bird_Opening_Wagner-Zwitersch_Gambit",
    "Bird_Opening_Williams-Zilbermints_Gambit",
    "Bird_Opening_Williams_Gambit",
    "Bishops_Opening",
    "Bishops_Opening_Anderssen_Gambit",
    "Bishops_Opening_Berlin_Defense",
    "Bishops_Opening_Boden-Kieseritzky_Gambit",
    "Bishops_Opening_Boi_Variation",
    "Bishops_Opening_Calabrese_Countergambit",
    "Bishops_Opening_Horwitz_Gambit",
    "Bishops_Opening_Khan_Gambit",
    "Bishops_Opening_Kitchener_Folly",
    "Bishops_Opening_Lewis_Countergambit",
    "Bishops_Opening_Lewis_Gambit",
    "Bishops_Opening_Lopez_Variation",
    "Bishops_Opening_McDonnell_Gambit",
    "Bishops_Opening_Other_variations",
    "Bishops_Opening_Pachman_Gambit",
    "Bishops_Opening_Philidor_Counterattack",
    "Bishops_Opening_Philidor_Variation",
    "Bishops_Opening_Ponziani_Gambit",
    "Bishops_Opening_Stein_Gambit",
    "Bishops_Opening_Urusov_Gambit",
    "Bishops_Opening_Vienna_Hybrid",
    "Bishops_Opening_Warsaw_Gambit",
    "Blackmar-Diemer_Gambit",
    "Blackmar-Diemer_Gambit_Accepted",
    "Blackmar-Diemer_Gambit_Accepted_Bogoljubow_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Euwe_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Gunderam_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Kaulich_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Other_variations",
    "Blackmar-Diemer_Gambit_Accepted_Pietrowsky_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Ritter_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Ryder_Gambit",
    "Blackmar-Diemer_Gambit_Accepted_Tartakower_Variation",
    "Blackmar-Diemer_Gambit_Accepted_Teichmann_Defense",
    "Blackmar-Diemer_Gambit_Accepted_Ziegler_Defense",
    "Blackmar-Diemer_Gambit_Blackmar_Gambit",
    "Blackmar-Diemer_Gambit_Bogoljubov_Variation",
    "Blackmar-Diemer_Gambit_Declined",
    "Blackmar-Diemer_Gambit_Declined_Brombacher_Countergambit",
    "Blackmar-Diemer_Gambit_Declined_Elbert_Countergambit",
    "Blackmar-Diemer_Gambit_Declined_Lamb_Defense",
    "Blackmar-Diemer_Gambit_Declined_Langeheinecke_Defense",
    "Blackmar-Diemer_Gambit_Declined_Langeheinicke_Defense",
    "Blackmar-Diemer_Gambit_Declined_OKelly_Defense",
    "Blackmar-Diemer_Gambit_Declined_Vienna_Defense",
    "Blackmar-Diemer_Gambit_Declined_Weinsbach_Declination",
    "Blackmar-Diemer_Gambit_Declined_Weinsbach_Defense",
    "Blackmar-Diemer_Gambit_Diemer-Rosenberg_Attack",
    "Blackmar-Diemer_Gambit_Euwe_Defense",
    "Blackmar-Diemer_Gambit_Fritz_Attack",
    "Blackmar-Diemer_Gambit_Gedult_Gambit",
    "Blackmar-Diemer_Gambit_Kaulich_Defense",
    "Blackmar-Diemer_Gambit_Lemberger_Countergambit",
    "Blackmar-Diemer_Gambit_Netherlands_Variation",
    "Blackmar-Diemer_Gambit_Other_variations",
    "Blackmar-Diemer_Gambit_Pietrowsky_Defense",
    "Blackmar-Diemer_Gambit_Rasa-Studier_Gambit",
    "Blackmar-Diemer_Gambit_Reversed_Albin_Countergambit",
    "Blackmar-Diemer_Gambit_Ritter_Defense",
    "Blackmar-Diemer_Gambit_Ryder_Gambit",
    "Blackmar-Diemer_Gambit_Tartakower_Variation",
    "Blackmar-Diemer_Gambit_Teichmann_Variation",
    "Blackmar-Diemer_Gambit_Vienna_Variation",
    "Blackmar-Diemer_Gambit_Zeller_Defense",
    "Blackmar-Diemer_Gambit_Ziegler_Defense",
    "Blackmar-Diemer_Gambit_von_Popiel_Gambit",
    "Blumenfeld_Countergambit",
    "Blumenfeld_Countergambit_Accepted",
    "Blumenfeld_Countergambit_Accepted_Other_variations",
    "Blumenfeld_Countergambit_Dus-Khotimirsky_Variation",
    "Blumenfeld_Countergambit_Other_variations",
    "Blumenfeld_Countergambit_Spielmann_Variation",
    "Bogo-Indian_Defense",
    "Bogo-Indian_Defense_Exchange_Variation",
    "Bogo-Indian_Defense_Grunfeld_Variation",
    "Bogo-Indian_Defense_Haiti_Variation",
    "Bogo-Indian_Defense_New_England_Variation",
    "Bogo-Indian_Defense_Nimzowitsch_Variation",
    "Bogo-Indian_Defense_Other_variations",
    "Bogo-Indian_Defense_Retreat_Variation",
    "Bogo-Indian_Defense_Vitolins_Variation",
    "Bogo-Indian_Defense_Wade-Smyslov_Variation",
    "Bongcloud_Attack",
    "Bongcloud_Attack_Other_variations",
    "Borg_Defense",
    "Borg_Defense_Borg_Gambit",
    "Borg_Defense_Other_variations",
    "Borg_Defense_Troon_Gambit",
    "Borg_Defense_Zilbermints_Gambit",
    "Borg_Opening",
    "Borg_Opening_Zilbermints_Gambit",
    "Bronstein_Gambit",
    "Bronstein_Gambit_Other_variations",
    "Canard_Opening",
    "Canard_Opening_Other_variations",
    "Caro-Kann_Defense",
    "Caro-Kann_Defense_Accelerated_Panov_Attack",
    "Caro-Kann_Defense_Advance",
    "Caro-Kann_Defense_Advance_Variation",
    "Caro-Kann_Defense_Alekhine_Gambit",
    "Caro-Kann_Defense_Alien_Gambit",
    "Caro-Kann_Defense_Apocalypse_Attack",
    "Caro-Kann_Defense_Breyer_Variation",
    "Caro-Kann_Defense_Bronstein-Larsen_Variation",
    "Caro-Kann_Defense_Campomanes_Attack",
    "Caro-Kann_Defense_Classical_Variation",
    "Caro-Kann_Defense_De_Bruycker_Defense",
    "Caro-Kann_Defense_Edinburgh_Variation",
    "Caro-Kann_Defense_Endgame_Offer",
    "Caro-Kann_Defense_Endgame_Variation",
    "Caro-Kann_Defense_Euwe_Attack",
    "Caro-Kann_Defense_Exchange_Variation",
    "Caro-Kann_Defense_Finnish_Variation",
    "Caro-Kann_Defense_Forgacs_Variation",
    "Caro-Kann_Defense_Goldman_Variation",
    "Caro-Kann_Defense_Gurgenidze_Counterattack",
    "Caro-Kann_Defense_Gurgenidze_System",
    "Caro-Kann_Defense_Hector_Gambit",
    "Caro-Kann_Defense_Hillbilly_Attack",
    "Caro-Kann_Defense_Karpov_Variation",
    "Caro-Kann_Defense_Labahn_Attack",
    "Caro-Kann_Defense_Main_Line",
    "Caro-Kann_Defense_Maroczy_Variation",
    "Caro-Kann_Defense_Masi_Variation",
    "Caro-Kann_Defense_Massachusetts_Defense",
    "Caro-Kann_Defense_Mieses_Gambit",
    "Caro-Kann_Defense_Modern_Variation",
    "Caro-Kann_Defense_Other_variations",
    "Caro-Kann_Defense_Panov-Botvinnik",
    "Caro-Kann_Defense_Panov_Attack",
    "Caro-Kann_Defense_Rasa-Studier_Gambit",
    "Caro-Kann_Defense_Scorpion-Horus_Gambit",
    "Caro-Kann_Defense_Spike_Variation",
    "Caro-Kann_Defense_St_Patricks_Attack",
    "Caro-Kann_Defense_Tartakower_Variation",
    "Caro-Kann_Defense_Toikkanen_Gambit",
    "Caro-Kann_Defense_Two_Knights_Attack",
    "Caro-Kann_Defense_Ulysses_Gambit",
    "Caro-Kann_Defense_von_Hennig_Gambit",
    "Carr_Defense",
    "Carr_Defense_Other_variations",
    "Carr_Defense_Zilbermints_Gambit",
    "Catalan_Opening",
    "Catalan_Opening_Closed",
    "Catalan_Opening_Closed_Variation",
    "Catalan_Opening_Hungarian_Gambit",
    "Catalan_Opening_Open_Defense",
    "Catalan_Opening_Other_variations",
    "Center_Game",
    "Center_Game_Accepted",
    "Center_Game_Accepted_Other_variations",
    "Center_Game_Berger_Variation",
    "Center_Game_Charousek_Variation",
    "Center_Game_Halasz-McDonnell_Gambit",
    "Center_Game_Hall_Variation",
    "Center_Game_Kieseritzky_Variation",
    "Center_Game_Lanc-Arnold_Gambit",
    "Center_Game_Normal_Variation",
    "Center_Game_Other_variations",
    "Center_Game_Paulsen_Attack_Variation",
    "Center_Game_Ross_Gambit",
    "Center_Game_lHermet_Variation",
    "Center_Game_von_der_Lasa_Gambit",
    "Clemenz_Opening",
    "Clemenz_Opening_Other_variations",
    "Colle_System",
    "Colle_System_Rhamphorhynchus_Variation",
    "Crab_Opening",
    "Crab_Opening_Other_variations",
    "Creepy_Crawly_Formation",
    "Creepy_Crawly_Formation_Classical_Defense",
    "Czech_Defense",
    "Czech_Defense_Other_variations",
    "Danish_Gambit",
    "Danish_Gambit_Accepted",
    "Danish_Gambit_Accepted_Chigorin_Defense",
    "Danish_Gambit_Accepted_Classical_Defense",
    "Danish_Gambit_Accepted_Copenhagen_Defense",
    "Danish_Gambit_Accepted_Other_variations",
    "Danish_Gambit_Accepted_Schlechter_Defense",
    "Danish_Gambit_Accepted_Svenonius_Defense",
    "Danish_Gambit_Declined",
    "Danish_Gambit_Declined_Sorensen_Defense",
    "Danish_Gambit_Other_variations",
    "Dory_Defense",
    "Dory_Defense_Other_variations",
    "Duras_Gambit",
    "Duras_Gambit_Other_variations",
    "Dutch_Defense",
    "Dutch_Defense_Alapin_Variation",
    "Dutch_Defense_Alekhine_Variation",
    "Dutch_Defense_Bellon_Gambit",
    "Dutch_Defense_Blackburne_Variation",
    "Dutch_Defense_Blackmars_Second_Gambit",
    "Dutch_Defense_Classical_Variation",
    "Dutch_Defense_Fianchetto_Attack",
    "Dutch_Defense_Fianchetto_Variation",
    "Dutch_Defense_Hevendehl_Gambit",
    "Dutch_Defense_Hopton_Attack",
    "Dutch_Defense_Hort-Antoshin_System",
    "Dutch_Defense_Janzen-Korchnoi_Gambit",
    "Dutch_Defense_Kingfisher_Gambit",
    "Dutch_Defense_Korchnoi_Attack",
    "Dutch_Defense_Krause_Variation",
    "Dutch_Defense_Krejcik_Gambit",
    "Dutch_Defense_Leningrad_Variation",
    "Dutch_Defense_Manhattan_Gambit",
    "Dutch_Defense_Nimzo-Dutch_Variation",
    "Dutch_Defense_Normal_Variation",
    "Dutch_Defense_Omega-Isis_Gambit",
    "Dutch_Defense_Other_variations",
    "Dutch_Defense_Queens_Knight_Variation",
    "Dutch_Defense_Raphael_Variation",
    "Dutch_Defense_Rubinstein_Variation",
    "Dutch_Defense_Semi-Leningrad_Variation",
    "Dutch_Defense_Spielmann_Gambit",
    "Dutch_Defense_Staunton_Gambit",
    "Dutch_Defense_Staunton_Gambit_Accepted",
    "Dutch_Defense_Stonewall",
    "Dutch_Defense_Stonewall_Variation",
    "East_Indian_Defense",
    "East_Indian_Defense_Other_variations",
    "Elephant_Gambit",
    "Elephant_Gambit_Maroczy_Gambit",
    "Elephant_Gambit_Other_variations",
    "Elephant_Gambit_Paulsen_Countergambit",
    "Elephant_Gambit_Wasp_Variation",
    "English_Defense",
    "English_Defense_Eastbourne_Gambit",
    "English_Defense_Hartlaub_Gambit_Accepted",
    "English_Defense_Hartlaub_Gambit_Declined",
    "English_Defense_Other_variations",
    "English_Defense_Perrin_Variation",
    "English_Defense_Poli_Gambit",
    "English_Opening",
    "English_Opening_Achilles-Omega_Gambit",
    "English_Opening_Adorjan_Defense",
    "English_Opening_Agincourt_Defense",
    "English_Opening_Anglo-Dutch_Defense",
    "English_Opening_Anglo-Dutch_Variation",
    "English_Opening_Anglo-Grunfeld_Defense",
    "English_Opening_Anglo-Indian_Defense",
    "English_Opening_Anglo-Lithuanian_Variation",
    "English_Opening_Anglo-Scandinavian_Defense",
    "English_Opening_Carls-Bremen_System",
    "English_Opening_Caro-Kann_Defensive_System",
    "English_Opening_Drill_Variation",
    "English_Opening_Four_Knights_System",
    "English_Opening_Great_Snake_Variation",
    "English_Opening_Jaenisch_Gambit",
    "English_Opening_Kings_English",
    "English_Opening_Kings_English_Variation",
    "English_Opening_Mikenas-Carls",
    "English_Opening_Mikenas-Carls_Variation",
    "English_Opening_Myers_Defense",
    "English_Opening_Myers_Gambit",
    "English_Opening_Neo-Catalan",
    "English_Opening_Neo-Catalan_Declined",
    "English_Opening_Other_variations",
    "English_Opening_Romanishin_Gambit",
    "English_Opening_Symmetrical",
    "English_Opening_Symmetrical_Variation",
    "English_Opening_The_Whale",
    "English_Opening_Wade_Gambit",
    "English_Opening_Wing_Gambit",
    "English_Orangutan",
    "English_Orangutan_Other_variations",
    "Englund_Gambit",
    "Englund_Gambit_Complex",
    "Englund_Gambit_Complex_Declined",
    "Englund_Gambit_Complex_Declined_Diemer_Counterattack",
    "Englund_Gambit_Complex_Declined_Other_variations",
    "Englund_Gambit_Complex_Englund_Gambit",
    "Englund_Gambit_Complex_Felbecker_Gambit",
    "Englund_Gambit_Complex_Hartlaub-Charlick_Gambit",
    "Englund_Gambit_Complex_Mosquito_Gambit",
    "Englund_Gambit_Complex_Soller_Gambit",
    "Englund_Gambit_Complex_Soller_Gambit_Deferred",
    "Englund_Gambit_Complex_Stockholm_Variation",
    "Englund_Gambit_Complex_Zilbermints_Gambit",
    "Englund_Gambit_Declined",
    "Englund_Gambit_Declined_Diemer_Counterattack",
    "Englund_Gambit_Declined_Other_variations",
    "Englund_Gambit_Declined_Reversed_Alekhine",
    "Englund_Gambit_Declined_Reversed_Brooklyn",
    "Englund_Gambit_Declined_Reversed_French",
    "Englund_Gambit_Declined_Reversed_Krebs",
    "Englund_Gambit_Declined_Reversed_Mokele_Mbembe",
    "Englund_Gambit_Felbecker_Gambit",
    "Englund_Gambit_Hartlaub-Charlick_Gambit",
    "Englund_Gambit_Main_Line",
    "Englund_Gambit_Mosquito_Gambit",
    "Englund_Gambit_Other_variations",
    "Englund_Gambit_Soller_Gambit",
    "Englund_Gambit_Soller_Gambit_Deferred",
    "Englund_Gambit_Stockholm_Variation",
    "Englund_Gambit_Zilbermints_Gambit",
    "Formation",
    "Formation_Shy_Attack",
    "Four_Knights_Game",
    "Four_Knights_Game_Bardeleben_Variation",
    "Four_Knights_Game_Double_Spanish",
    "Four_Knights_Game_Glek_System",
    "Four_Knights_Game_Gunsberg_Counterattack",
    "Four_Knights_Game_Gunsberg_Variation",
    "Four_Knights_Game_Halloween_Gambit",
    "Four_Knights_Game_Italian_Variation",
    "Four_Knights_Game_Marshall_Variation",
    "Four_Knights_Game_Nimzowitsch_Variation",
    "Four_Knights_Game_Other_variations",
    "Four_Knights_Game_Ranken_Variation",
    "Four_Knights_Game_Rubinstein_Countergambit",
    "Four_Knights_Game_Scotch_Variation",
    "Four_Knights_Game_Scotch_Variation_Accepted",
    "Four_Knights_Game_Spanish_Variation",
    "Four_Knights_Game_Spielmann_Variation",
    "Four_Knights_Game_Symmetrical",
    "French_Defense",
    "French_Defense_Advance_Variation",
    "French_Defense_Alapin_Gambit",
    "French_Defense_Alekhine-Chatard_Attack",
    "French_Defense_Baeuerle_Gambit",
    "French_Defense_Banzai-Leong_Gambit",
    "French_Defense_Bird_Invitation",
    "French_Defense_Burn_Variation",
    "French_Defense_Carlson_Gambit",
    "French_Defense_Chigorin_Variation",
    "French_Defense_Classical_Variation",
    "French_Defense_Diemer-Duhm_Gambit",
    "French_Defense_Diemer-Duhm_Gambit_Accepted",
    "French_Defense_Exchange_Variation",
    "French_Defense_Franco-Hiva_Gambit",
    "French_Defense_Franco-Hiva_Gambit_Accepted",
    "French_Defense_Franco-Sicilian_Defense",
    "French_Defense_Henneberger_Variation",
    "French_Defense_Horwitz_Attack",
    "French_Defense_Kings_Indian_Attack",
    "French_Defense_Knight_Variation",
    "French_Defense_La_Bourdonnais_Variation",
    "French_Defense_MacCutcheon_Variation",
    "French_Defense_Mediterranean_Defense",
    "French_Defense_Morphy_Gambit",
    "French_Defense_Normal_Variation",
    "French_Defense_Orthoschnapp_Gambit",
    "French_Defense_Other_variations",
    "French_Defense_Paulsen_Variation",
    "French_Defense_Pelikan_Variation",
    "French_Defense_Perseus_Gambit",
    "French_Defense_Queens_Knight",
    "French_Defense_Reti-Spielmann_Attack",
    "French_Defense_Reversed_Philidor_Formation",
    "French_Defense_Rubinstein_Variation",
    "French_Defense_Schlechter_Variation",
    "French_Defense_St_George_Defense",
    "French_Defense_Steiner_Variation",
    "French_Defense_Steinitz_Attack",
    "French_Defense_Steinitz_Variation",
    "French_Defense_Tarrasch_Variation",
    "French_Defense_Two_Knights_Variation",
    "French_Defense_Winawer_Variation",
    "French_Defense_Wing_Gambit",
    "Fried_Fox_Defense",
    "Fried_Fox_Defense_Other_variations",
    "Gedults_Opening",
    "Gedults_Opening_Other_variations",
    "Giuoco_Piano",
    "Giuoco_Piano_Other_variations",
    "Global_Opening",
    "Global_Opening_Other_variations",
    "Goldsmith_Defense",
    "Goldsmith_Defense_Other_variations",
    "Goldsmith_Defense_Picklepuss_Defense",
    "Grob_Opening",
    "Grob_Opening_Alessi_Gambit",
    "Grob_Opening_Double_Grob",
    "Grob_Opening_Grob_Gambit",
    "Grob_Opening_Grob_Gambit_Declined",
    "Grob_Opening_Keene_Defense",
    "Grob_Opening_London_Defense",
    "Grob_Opening_Other_variations",
    "Grob_Opening_Romford_Countergambit",
    "Grob_Opening_Spike",
    "Grob_Opening_Spike_Attack",
    "Grob_Opening_Zilbermints_Gambit",
    "Grunfeld_Defense",
    "Grunfeld_Defense_Botvinnik_Variation",
    "Grunfeld_Defense_Brinckmann_Attack",
    "Grunfeld_Defense_Counterthrust_Variation",
    "Grunfeld_Defense_Exchange_Variation",
    "Grunfeld_Defense_Flohr_Variation",
    "Grunfeld_Defense_Lundin_Variation",
    "Grunfeld_Defense_Lutikov_Variation",
    "Grunfeld_Defense_Makogonov_Variation",
    "Grunfeld_Defense_Opocensky_Variation",
    "Grunfeld_Defense_Other_variations",
    "Grunfeld_Defense_Pachman_Variation",
    "Grunfeld_Defense_Russian_Variation",
    "Grunfeld_Defense_Smyslov_Defense",
    "Grunfeld_Defense_Stockholm_Variation",
    "Grunfeld_Defense_Three_Knights_Variation",
    "Grunfeld_Defense_Zaitsev_Gambit",
    "Guatemala_Defense",
    "Guatemala_Defense_Other_variations",
    "Gunderam_Defense",
    "Gunderam_Defense_Other_variations",
    "Hippopotamus_Defense",
    "Hippopotamus_Defense_Other_variations",
    "Horwitz_Defense",
    "Horwitz_Defense_Other_variations",
    "Hungarian_Opening",
    "Hungarian_Opening_Bucker_Gambit",
    "Hungarian_Opening_Catalan_Formation",
    "Hungarian_Opening_Dutch_Defense",
    "Hungarian_Opening_Indian_Defense",
    "Hungarian_Opening_Myers_Defense",
    "Hungarian_Opening_Other_variations",
    "Hungarian_Opening_Reversed_Alekhine",
    "Hungarian_Opening_Reversed_Modern_Defense",
    "Hungarian_Opening_Reversed_Norwegian_Defense",
    "Hungarian_Opening_Sicilian_Invitation",
    "Hungarian_Opening_Slav_Formation",
    "Hungarian_Opening_Symmetrical_Variation",
    "Hungarian_Opening_Van_Kuijk_Gambit",
    "Indian_Defense",
    "Indian_Defense_Anti-Grunfeld",
    "Indian_Defense_Anti-Nimzo-Indian",
    "Indian_Defense_Budapest_Defense",
    "Indian_Defense_Colle_System",
    "Indian_Defense_Czech-Indian",
    "Indian_Defense_Devin_Gambit",
    "Indian_Defense_Dory_Indian",
    "Indian_Defense_Dzindzi-Indian_Defense",
    "Indian_Defense_Gedult_Attack",
    "Indian_Defense_Gibbins-Weidenhagen_Gambit",
    "Indian_Defense_Gibbins-Weidenhagen_Gambit_Accepted",
    "Indian_Defense_Gibbins-Wiedenhagen_Gambit",
    "Indian_Defense_Gibbins-Wiedenhagen_Gambit_Accepted",
    "Indian_Defense_Kings_Indian_Variation",
    "Indian_Defense_Knights_Variation",
    "Indian_Defense_Lazard_Gambit",
    "Indian_Defense_London_System",
    "Indian_Defense_Maddigan_Gambit",
    "Indian_Defense_Medusa_Gambit",
    "Indian_Defense_Normal_Variation",
    "Indian_Defense_Omega_Gambit",
    "Indian_Defense_Other_variations",
    "Indian_Defense_Paleface_Attack",
    "Indian_Defense_Pawn_Push_Variation",
    "Indian_Defense_Polish_Variation",
    "Indian_Defense_Przepiorka_Variation",
    "Indian_Defense_Pseudo-Benko",
    "Indian_Defense_Pyrenees_Gambit",
    "Indian_Defense_Reversed_Chigorin_Defense",
    "Indian_Defense_Seirawan_Attack",
    "Indian_Defense_Spielmann-Indian",
    "Indian_Defense_Tartakower_Attack",
    "Indian_Defense_Wade-Tartakower_Defense",
    "Indian_Defense_West_Indian_Defense",
    "Irish_Gambit",
    "Irish_Gambit_Other_variations",
    "Italian_Game",
    "Italian_Game_Anti-Fried_Liver_Defense",
    "Italian_Game_Birds_Attack",
    "Italian_Game_Blackburne-Kostic_Gambit",
    "Italian_Game_Classical_Variation",
    "Italian_Game_Deutz_Gambit",
    "Italian_Game_Evans_Gambit",
    "Italian_Game_Evans_Gambit_Accepted",
    "Italian_Game_Evans_Gambit_Declined",
    "Italian_Game_Giuoco_Pianissimo",
    "Italian_Game_Giuoco_Piano",
    "Italian_Game_Hungarian_Defense",
    "Italian_Game_Jerome_Gambit",
    "Italian_Game_Other_variations",
    "Italian_Game_Paris_Defense",
    "Italian_Game_Rosentreter_Gambit",
    "Italian_Game_Rousseau_Gambit",
    "Italian_Game_Schilling-Kostic_Gambit",
    "Italian_Game_Scotch_Gambit",
    "Italian_Game_Scotch_Gambit_Declined",
    "Italian_Game_Scotch_Invitation_Declined",
    "Italian_Game_Two_Knights_Defense",
    "Kadas_Opening",
    "Kadas_Opening_Beginners_Trap",
    "Kadas_Opening_Kadas_Gambit",
    "Kadas_Opening_Myers_Variation",
    "Kadas_Opening_Other_variations",
    "Kadas_Opening_Schneider_Gambit",
    "Kangaroo_Defense",
    "Kangaroo_Defense_Keres_Defense",
    "Kangaroo_Defense_Other_variations",
    "Kings_Gambit",
    "Kings_Gambit_Accepted",
    "Kings_Gambit_Accepted_Abbazia_Defense",
    "Kings_Gambit_Accepted_Allgaier",
    "Kings_Gambit_Accepted_Allgaier_Gambit",
    "Kings_Gambit_Accepted_Basman_Gambit",
    "Kings_Gambit_Accepted_Becker_Defense",
    "Kings_Gambit_Accepted_Bishops_Gambit",
    "Kings_Gambit_Accepted_Blachly_Gambit",
    "Kings_Gambit_Accepted_Bonsch-Osmolovsky_Variation",
    "Kings_Gambit_Accepted_Breyer_Gambit",
    "Kings_Gambit_Accepted_Carrera_Gambit",
    "Kings_Gambit_Accepted_Cunningham_Defense",
    "Kings_Gambit_Accepted_Dodo_Variation",
    "Kings_Gambit_Accepted_Double_Muzio_Gambit",
    "Kings_Gambit_Accepted_Eisenberg_Variation",
    "Kings_Gambit_Accepted_Fischer_Defense",
    "Kings_Gambit_Accepted_Gaga_Gambit",
    "Kings_Gambit_Accepted_Ghulam-Kassim_Gambit",
    "Kings_Gambit_Accepted_Gianutio_Countergambit",
    "Kings_Gambit_Accepted_Greco_Gambit",
    "Kings_Gambit_Accepted_Hanstein_Gambit",
    "Kings_Gambit_Accepted_Kieseritzky",
    "Kings_Gambit_Accepted_Kieseritzky_Gambit",
    "Kings_Gambit_Accepted_Kings_Knights_Gambit",
    "Kings_Gambit_Accepted_Kotov_Gambit",
    "Kings_Gambit_Accepted_Lolli_Gambit",
    "Kings_Gambit_Accepted_MacLeod_Defense",
    "Kings_Gambit_Accepted_Mason-Keres_Gambit",
    "Kings_Gambit_Accepted_Mayet_Gambit",
    "Kings_Gambit_Accepted_McDonnell_Gambit",
    "Kings_Gambit_Accepted_Modern_Defense",
    "Kings_Gambit_Accepted_Muzio_Gambit",
    "Kings_Gambit_Accepted_Muzio_Gambit_Accepted",
    "Kings_Gambit_Accepted_Other_variations",
    "Kings_Gambit_Accepted_Paris_Gambit",
    "Kings_Gambit_Accepted_Philidor_Gambit",
    "Kings_Gambit_Accepted_Polerio_Gambit",
    "Kings_Gambit_Accepted_Quaade_Gambit",
    "Kings_Gambit_Accepted_Quade_Gambit",
    "Kings_Gambit_Accepted_Rosentreter_Gambit",
    "Kings_Gambit_Accepted_Salvio_Gambit",
    "Kings_Gambit_Accepted_Schallopp_Defense",
    "Kings_Gambit_Accepted_Silberschmidt_Gambit",
    "Kings_Gambit_Accepted_Sorensen_Gambit",
    "Kings_Gambit_Accepted_Stamma_Gambit",
    "Kings_Gambit_Accepted_Tartakower_Gambit",
    "Kings_Gambit_Accepted_Traditional_Variation",
    "Kings_Gambit_Accepted_Tumbleweed",
    "Kings_Gambit_Accepted_Villemson_Gambit",
    "Kings_Gambit_Accepted_Wagenbach_Defense",
    "Kings_Gambit_Declined",
    "Kings_Gambit_Declined_Classical",
    "Kings_Gambit_Declined_Classical_Variation",
    "Kings_Gambit_Declined_Falkbeer_Countergambit",
    "Kings_Gambit_Declined_Falkbeer_Countergambit_Accepted",
    "Kings_Gambit_Declined_Keene_Defense",
    "Kings_Gambit_Declined_Keenes_Defense",
    "Kings_Gambit_Declined_Mafia_Defense",
    "Kings_Gambit_Declined_Miles_Defense",
    "Kings_Gambit_Declined_Norwalde_Variation",
    "Kings_Gambit_Declined_Panteldakis_Countergambit",
    "Kings_Gambit_Declined_Petrovs_Defense",
    "Kings_Gambit_Declined_Queens_Knight_Defense",
    "Kings_Gambit_Declined_Soller-Zilbermints_Gambit",
    "Kings_Gambit_Declined_Zilbermints_Double_Countergambit",
    "Kings_Gambit_Other_variations",
    "Kings_Indian_Attack",
    "Kings_Indian_Attack_Double_Fianchetto",
    "Kings_Indian_Attack_French_Variation",
    "Kings_Indian_Attack_Keres_Variation",
    "Kings_Indian_Attack_Omega-Delta_Gambit",
    "Kings_Indian_Attack_Other_variations",
    "Kings_Indian_Attack_Pachman_System",
    "Kings_Indian_Attack_Sicilian_Variation",
    "Kings_Indian_Attack_Smyslov_Variation",
    "Kings_Indian_Attack_Spassky_Variation",
    "Kings_Indian_Attack_Symmetrical_Defense",
    "Kings_Indian_Attack_Wahls_Defense",
    "Kings_Indian_Attack_Yugoslav_Variation",
    "Kings_Indian_Defense",
    "Kings_Indian_Defense_Accelerated_Averbakh_Variation",
    "Kings_Indian_Defense_Averbakh_Variation",
    "Kings_Indian_Defense_Exchange_Variation",
    "Kings_Indian_Defense_Fianchetto_Variation",
    "Kings_Indian_Defense_Four_Pawns_Attack",
    "Kings_Indian_Defense_Kazakh_Variation",
    "Kings_Indian_Defense_Kramer_Variation",
    "Kings_Indian_Defense_Larsen_Variation",
    "Kings_Indian_Defense_Makogonov_Variation",
    "Kings_Indian_Defense_Normal_Variation",
    "Kings_Indian_Defense_Orthodox_Variation",
    "Kings_Indian_Defense_Other_variations",
    "Kings_Indian_Defense_Petrosian_Variation",
    "Kings_Indian_Defense_Pomar_System",
    "Kings_Indian_Defense_Samisch_Variation",
    "Kings_Indian_Defense_Santasiere_Variation",
    "Kings_Indian_Defense_Semi-Averbakh_System",
    "Kings_Indian_Defense_Semi-Classical_Variation",
    "Kings_Indian_Defense_Smyslov_Variation",
    "Kings_Indian_Defense_Steiner_Attack",
    "Kings_Indian_Defense_Zinnowitz_Variation",
    "Kings_Knight_Opening",
    "Kings_Knight_Opening_Konstantinopolsky",
    "Kings_Knight_Opening_Normal_Variation",
    "Kings_Knight_Opening_Other_variations",
    "Kings_Pawn",
    "Kings_Pawn_Game",
    "Kings_Pawn_Game_Alapin_Opening",
    "Kings_Pawn_Game_Bavarian_Gambit",
    "Kings_Pawn_Game_Beyer_Gambit",
    "Kings_Pawn_Game_Busch-Gass_Gambit",
    "Kings_Pawn_Game_Clam_Variation",
    "Kings_Pawn_Game_Damiano_Defense",
    "Kings_Pawn_Game_Dresden_Opening",
    "Kings_Pawn_Game_Gunderam_Gambit",
    "Kings_Pawn_Game_Kings_Head_Opening",
    "Kings_Pawn_Game_La_Bourdonnais_Gambit",
    "Kings_Pawn_Game_Leonardis_Variation",
    "Kings_Pawn_Game_MacLeod_Attack",
    "Kings_Pawn_Game_Macleod_Attack",
    "Kings_Pawn_Game_Maroczy_Defense",
    "Kings_Pawn_Game_McConnell_Defense",
    "Kings_Pawn_Game_Mengarinis_Opening",
    "Kings_Pawn_Game_Napoleon_Attack",
    "Kings_Pawn_Game_Other_variations",
    "Kings_Pawn_Game_Pachman_Wing_Gambit",
    "Kings_Pawn_Game_Philidor_Gambit",
    "Kings_Pawn_Game_Schulze-Muller_Gambit",
    "Kings_Pawn_Game_Tayler_Opening",
    "Kings_Pawn_Game_Tortoise_Opening",
    "Kings_Pawn_Game_Wayward_Queen_Attack",
    "Kings_Pawn_Game_Weber_Gambit",
    "Kings_Pawn_Opening",
    "Kings_Pawn_Opening_Other_variations",
    "Kings_Pawn_Opening_Speers",
    "Kings_Pawn_Opening_Van_Hooydoon_Gambit",
    "Kings_Pawn_Other_variations",
    "Lasker_Simul_Special",
    "Lasker_Simul_Special_Other_variations",
    "Latvian_Gambit",
    "Latvian_Gambit_Accepted",
    "Latvian_Gambit_Accepted_Bilguer_Variation",
    "Latvian_Gambit_Accepted_Bronstein_Attack",
    "Latvian_Gambit_Accepted_Bronstein_Gambit",
    "Latvian_Gambit_Accepted_Foltys-Leonhardt_Variation",
    "Latvian_Gambit_Accepted_Foltys_Variation",
    "Latvian_Gambit_Accepted_Leonhardt_Variation",
    "Latvian_Gambit_Accepted_Main_Line",
    "Latvian_Gambit_Accepted_Nimzowitsch_Attack",
    "Latvian_Gambit_Accepted_Other_variations",
    "Latvian_Gambit_Clam_Gambit",
    "Latvian_Gambit_Corkscrew_Countergambit",
    "Latvian_Gambit_Corkscrew_Gambit",
    "Latvian_Gambit_Fraser_Defense",
    "Latvian_Gambit_Greco_Variation",
    "Latvian_Gambit_Mason_Countergambit",
    "Latvian_Gambit_Mayet_Attack",
    "Latvian_Gambit_Mlotkowski_Variation",
    "Latvian_Gambit_Other_variations",
    "Lemming_Defense",
    "Lemming_Defense_Other_variations",
    "Lion_Defense",
    "Lion_Defense_Anti-Philidor",
    "Lion_Defense_Bayonet_Attack",
    "Lion_Defense_Lions_Jaw",
    "London_System",
    "London_System_Other_variations",
    "London_System_Poisoned_Pawn_Variation",
    "Marienbad_System",
    "Marienbad_System_Other_variations",
    "Mexican_Defense",
    "Mexican_Defense_Horsefly_Gambit",
    "Mexican_Defense_Other_variations",
    "Mieses_Opening",
    "Mieses_Opening_Myers_Spike_Attack",
    "Mieses_Opening_Other_variations",
    "Mieses_Opening_Reversed_Rat",
    "Mikenas_Defense",
    "Mikenas_Defense_Cannstatter_Variation",
    "Mikenas_Defense_Lithuanian_Variation",
    "Mikenas_Defense_Other_variations",
    "Mikenas_Defense_Pozarek_Gambit",
    "Modern_Defense",
    "Modern_Defense_Averbakh_System",
    "Modern_Defense_Averbakh_Variation",
    "Modern_Defense_Beefeater_Variation",
    "Modern_Defense_Bishop_Attack",
    "Modern_Defense_Fianchetto_Gambit",
    "Modern_Defense_Gellers_System",
    "Modern_Defense_Gurgenidze_Defense",
    "Modern_Defense_Kotov_Variation",
    "Modern_Defense_Lizard_Defense",
    "Modern_Defense_Modern_Pterodactyl",
    "Modern_Defense_Mongredien_Defense",
    "Modern_Defense_Neo-Modern_Defense",
    "Modern_Defense_Norwegian_Defense",
    "Modern_Defense_Other_variations",
    "Modern_Defense_Pseudo-Austrian_Attack",
    "Modern_Defense_Pterodactyl_Variation",
    "Modern_Defense_Randspringer_Variation",
    "Modern_Defense_Semi-Averbakh_Variation",
    "Modern_Defense_Standard_Defense",
    "Modern_Defense_Standard_Line",
    "Modern_Defense_Three_Pawns_Attack",
    "Modern_Defense_Two_Knights_Variation",
    "Modern_Defense_Westermann_Gambit",
    "Modern_Defense_Wind_Gambit",
    "Montevideo_Defense",
    "Montevideo_Defense_Other_variations",
    "Neo-Grunfeld_Defense",
    "Neo-Grunfeld_Defense_Classical_Variation",
    "Neo-Grunfeld_Defense_Delayed_Exchange_Variation",
    "Neo-Grunfeld_Defense_Exchange_Variation",
    "Neo-Grunfeld_Defense_Goglidze_Attack",
    "Neo-Grunfeld_Defense_Non-_or_Delayed_Fianchetto",
    "Neo-Grunfeld_Defense_Other_variations",
    "Neo-Grunfeld_Defense_Ultra-Delayed_Exchange_Variation",
    "Neo-Grunfeld_Defense_Ultra-delayed_Exchange_Variation",
    "Neo-Grunfeld_Defense_with_Nf3",
    "Neo-Grunfeld_Defense_with_g3",
    "Nimzo-Indian_Defense",
    "Nimzo-Indian_Defense_Classical_Variation",
    "Nimzo-Indian_Defense_Fischer_Variation",
    "Nimzo-Indian_Defense_Hubner_Variation",
    "Nimzo-Indian_Defense_Kmoch_Variation",
    "Nimzo-Indian_Defense_Leningrad_Variation",
    "Nimzo-Indian_Defense_Mikenas_Attack",
    "Nimzo-Indian_Defense_Normal_Line",
    "Nimzo-Indian_Defense_Normal_Variation",
    "Nimzo-Indian_Defense_Other_variations",
    "Nimzo-Indian_Defense_Ragozin_Defense",
    "Nimzo-Indian_Defense_Ragozin_Variation",
    "Nimzo-Indian_Defense_Reshevsky_Variation",
    "Nimzo-Indian_Defense_Romanishin_Variation",
    "Nimzo-Indian_Defense_Samisch_Variation",
    "Nimzo-Indian_Defense_Simagin_Variation",
    "Nimzo-Indian_Defense_Spielmann_Variation",
    "Nimzo-Indian_Defense_St_Petersburg_Variation",
    "Nimzo-Indian_Defense_Three_Knights_Variation",
    "Nimzo-Larsen_Attack",
    "Nimzo-Larsen_Attack_Classical_Variation",
    "Nimzo-Larsen_Attack_Dutch_Variation",
    "Nimzo-Larsen_Attack_English_Variation",
    "Nimzo-Larsen_Attack_Graz_Attack",
    "Nimzo-Larsen_Attack_Indian_Variation",
    "Nimzo-Larsen_Attack_Modern_Variation",
    "Nimzo-Larsen_Attack_Norfolk_Gambit",
    "Nimzo-Larsen_Attack_Other_variations",
    "Nimzo-Larsen_Attack_Pachman_Gambit",
    "Nimzo-Larsen_Attack_Polish_Variation",
    "Nimzo-Larsen_Attack_Ringelbach_Gambit",
    "Nimzo-Larsen_Attack_Spike_Variation",
    "Nimzo-Larsen_Attack_Symmetrical_Variation",
    "Nimzowitsch_Defense",
    "Nimzowitsch_Defense_Breyer_Variation",
    "Nimzowitsch_Defense_Colorado_Countergambit",
    "Nimzowitsch_Defense_Colorado_Countergambit_Accepted",
    "Nimzowitsch_Defense_Declined_Variation",
    "Nimzowitsch_Defense_El_Columpio_Defense",
    "Nimzowitsch_Defense_Franco-Nimzowitsch_Variation",
    "Nimzowitsch_Defense_French_Connection",
    "Nimzowitsch_Defense_Hornung_Gambit",
    "Nimzowitsch_Defense_Kennedy_Variation",
    "Nimzowitsch_Defense_Mikenas_Variation",
    "Nimzowitsch_Defense_Neo-Mongoloid_Defense",
    "Nimzowitsch_Defense_Other_variations",
    "Nimzowitsch_Defense_Pirc_Connection",
    "Nimzowitsch_Defense_Pseudo-Spanish_Variation",
    "Nimzowitsch_Defense_Scandinavian_Variation",
    "Nimzowitsch_Defense_Williams_Variation",
    "Nimzowitsch_Defense_Woodchuck_Variation",
    "Norwegian_Defense",
    "Norwegian_Defense_Other_variations",
    "Old_Indian_Defense",
    "Old_Indian_Defense_Czech_Variation",
    "Old_Indian_Defense_Dus-Khotimirsky_Variation",
    "Old_Indian_Defense_Janowski_Variation",
    "Old_Indian_Defense_Normal_Variation",
    "Old_Indian_Defense_Other_variations",
    "Old_Indian_Defense_Tartakower-Indian",
    "Old_Indian_Defense_Two_Knights_Variation",
    "Old_Indian_Defense_Ukrainian_Variation",
    "Owen_Defense",
    "Owen_Defense_Hekili-Loa_Gambit",
    "Owen_Defense_Matovinsky_Gambit",
    "Owen_Defense_Naselwaus_Gambit",
    "Owen_Defense_Other_variations",
    "Owen_Defense_Smith_Gambit",
    "Owen_Defense_Wind_Gambit",
    "Paleface_Attack",
    "Paleface_Attack_Other_variations",
    "Petrovs_Defense",
    "Petrovs_Defense_Classical_Attack",
    "Petrovs_Defense_Cochrane_Gambit",
    "Petrovs_Defense_Cozio_Attack",
    "Petrovs_Defense_Damiano_Variation",
    "Petrovs_Defense_French_Attack",
    "Petrovs_Defense_Italian_Variation",
    "Petrovs_Defense_Karklins-Martinovsky_Variation",
    "Petrovs_Defense_Kaufmann_Attack",
    "Petrovs_Defense_Millennium_Attack",
    "Petrovs_Defense_Modern_Attack",
    "Petrovs_Defense_Nimzowitsch_Attack",
    "Petrovs_Defense_Other_variations",
    "Petrovs_Defense_Paulsen_Attack",
    "Petrovs_Defense_Stafford_Gambit",
    "Petrovs_Defense_Three_Knights_Game",
    "Philidor_Defense",
    "Philidor_Defense_Albin-Blackburne_Gambit",
    "Philidor_Defense_Bird_Gambit",
    "Philidor_Defense_Boden_Variation",
    "Philidor_Defense_Exchange_Variation",
    "Philidor_Defense_Hanham",
    "Philidor_Defense_Hanham_Variation",
    "Philidor_Defense_Larsen_Variation",
    "Philidor_Defense_Lion_Variation",
    "Philidor_Defense_Lopez_Countergambit",
    "Philidor_Defense_Morphy_Gambit",
    "Philidor_Defense_Nimzowitsch",
    "Philidor_Defense_Nimzowitsch_Variation",
    "Philidor_Defense_Other_variations",
    "Philidor_Defense_Paulsen_Attack",
    "Philidor_Defense_Philidor_Countergambit",
    "Philidor_Defense_Philidor_Gambit",
    "Philidor_Defense_Steinitz_Variation",
    "Pirc_Defense",
    "Pirc_Defense_150_Attack",
    "Pirc_Defense_Austrian_Attack",
    "Pirc_Defense_Bayonet_Attack",
    "Pirc_Defense_Byrne_Variation",
    "Pirc_Defense_Chinese_Variation",
    "Pirc_Defense_Classical_Variation",
    "Pirc_Defense_Kholmov_System",
    "Pirc_Defense_Other_variations",
    "Pirc_Defense_Roscher_Gambit",
    "Pirc_Defense_Sveshnikov_System",
    "Polish_Defense",
    "Polish_Defense_Other_variations",
    "Polish_Defense_Spassky_Gambit_Accepted",
    "Polish_Opening",
    "Polish_Opening_Baltic_Defense",
    "Polish_Opening_Birmingham_Gambit",
    "Polish_Opening_Bugayev_Advance_Variation",
    "Polish_Opening_Bugayev_Attack",
    "Polish_Opening_Czech_Defense",
    "Polish_Opening_Dutch_Defense",
    "Polish_Opening_German_Defense",
    "Polish_Opening_Grigorian_Variation",
    "Polish_Opening_Kings_Indian_Variation",
    "Polish_Opening_Myers_Variation",
    "Polish_Opening_Other_variations",
    "Polish_Opening_Outflank_Variation",
    "Polish_Opening_Queens_Indian_Variation",
    "Polish_Opening_Queenside_Defense",
    "Polish_Opening_Rooks_Swap_Line",
    "Polish_Opening_Schiffler-Sokolsky_Variation",
    "Polish_Opening_Schuehler_Gambit",
    "Polish_Opening_Symmetrical_Variation",
    "Polish_Opening_Tartakower_Gambit",
    "Polish_Opening_Wolferts_Gambit",
    "Polish_Opening_Zukertort_System",
    "Ponziani_Opening",
    "Ponziani_Opening_Caro_Gambit",
    "Ponziani_Opening_Jaenisch_Counterattack",
    "Ponziani_Opening_Leonhardt_Variation",
    "Ponziani_Opening_Neumann_Gambit",
    "Ponziani_Opening_Other_variations",
    "Ponziani_Opening_Ponziani_Countergambit",
    "Ponziani_Opening_Reti_Variation",
    "Ponziani_Opening_Romanishin_Variation",
    "Ponziani_Opening_Spanish_Variation",
    "Ponziani_Opening_Steinitz_Variation",
    "Ponziani_Opening_Vukovic_Gambit",
    "Portuguese_Opening",
    "Portuguese_Opening_Other_variations",
    "Portuguese_Opening_Portuguese_Gambit",
    "Pseudo_Queens_Indian_Defense",
    "Pseudo_Queens_Indian_Defense_Other_variations",
    "Pterodactyl_Defense",
    "Pterodactyl_Defense_Austrian",
    "Pterodactyl_Defense_Central",
    "Pterodactyl_Defense_Eastern",
    "Pterodactyl_Defense_Fianchetto",
    "Pterodactyl_Defense_Other_variations",
    "Pterodactyl_Defense_Queen_Pterodactyl",
    "Pterodactyl_Defense_Sicilian",
    "Pterodactyl_Defense_Western",
    "Queens_Gambit",
    "Queens_Gambit_Accepted",
    "Queens_Gambit_Accepted_Accelerated_Mannheim_Variation",
    "Queens_Gambit_Accepted_Alekhine_Defense",
    "Queens_Gambit_Accepted_Bogoljubov_Defense",
    "Queens_Gambit_Accepted_Bogoljubow_Defense",
    "Queens_Gambit_Accepted_Central_Variation",
    "Queens_Gambit_Accepted_Classical_Defense",
    "Queens_Gambit_Accepted_Furman_Variation",
    "Queens_Gambit_Accepted_Godes_Variation",
    "Queens_Gambit_Accepted_Gunsberg_Defense",
    "Queens_Gambit_Accepted_Janowski-Larsen_Variation",
    "Queens_Gambit_Accepted_Linares_Variation",
    "Queens_Gambit_Accepted_Mannheim_Variation",
    "Queens_Gambit_Accepted_Normal_Variation",
    "Queens_Gambit_Accepted_Old_Variation",
    "Queens_Gambit_Accepted_Other_variations",
    "Queens_Gambit_Accepted_Rosenthal_Variation",
    "Queens_Gambit_Accepted_Saduleto_Variation",
    "Queens_Gambit_Accepted_Schwartz_Defense",
    "Queens_Gambit_Accepted_Showalter_Variation",
    "Queens_Gambit_Accepted_Slav_Gambit",
    "Queens_Gambit_Accepted_Smyslov_Variation",
    "Queens_Gambit_Accepted_Winawer_Defense",
    "Queens_Gambit_Declined",
    "Queens_Gambit_Declined_Alapin_Variation",
    "Queens_Gambit_Declined_Albin_Countergambit",
    "Queens_Gambit_Declined_Alekhine_Variation",
    "Queens_Gambit_Declined_Anti-Tartakower_Variation",
    "Queens_Gambit_Declined_Austrian_Attack",
    "Queens_Gambit_Declined_Austrian_Defense",
    "Queens_Gambit_Declined_Baltic_Defense",
    "Queens_Gambit_Declined_Barmen_Variation",
    "Queens_Gambit_Declined_Been-Koomen_Variation",
    "Queens_Gambit_Declined_Cambridge_Springs_Defense",
    "Queens_Gambit_Declined_Capablanca_Variation",
    "Queens_Gambit_Declined_Charousek_Variation",
    "Queens_Gambit_Declined_Chigorin_Defense",
    "Queens_Gambit_Declined_Exchange_Variation",
    "Queens_Gambit_Declined_Harrwitz_Attack",
    "Queens_Gambit_Declined_Janowski_Variation",
    "Queens_Gambit_Declined_Knight_Defense",
    "Queens_Gambit_Declined_Lasker_Defense",
    "Queens_Gambit_Declined_Manhattan_Variation",
    "Queens_Gambit_Declined_Marshall_Defense",
    "Queens_Gambit_Declined_Miles_Variation",
    "Queens_Gambit_Declined_Modern_Variation",
    "Queens_Gambit_Declined_Neo-Orthodox_Variation",
    "Queens_Gambit_Declined_Normal_Defense",
    "Queens_Gambit_Declined_Orthodox_Defense",
    "Queens_Gambit_Declined_Other_variations",
    "Queens_Gambit_Declined_Pillsbury_Attack",
    "Queens_Gambit_Declined_Pseudo-Tarrasch_Variation",
    "Queens_Gambit_Declined_Queens_Knight_Variation",
    "Queens_Gambit_Declined_Ragozin_Defense",
    "Queens_Gambit_Declined_Semi-Slav",
    "Queens_Gambit_Declined_Semi-Tarrasch_Defense",
    "Queens_Gambit_Declined_Semmering_Variation",
    "Queens_Gambit_Declined_Spielmann_Variation",
    "Queens_Gambit_Declined_Tarrasch_Defense",
    "Queens_Gambit_Declined_Tartakower_Defense",
    "Queens_Gambit_Declined_Three_Knights",
    "Queens_Gambit_Declined_Three_Knights_Variation",
    "Queens_Gambit_Declined_Traditional_Variation",
    "Queens_Gambit_Declined_Vienna_Variation",
    "Queens_Gambit_Declined_Westphalian_Variation",
    "Queens_Gambit_Declined_Zilbermints_Gambit",
    "Queens_Gambit_Other_variations",
    "Queens_Indian_Accelerated",
    "Queens_Indian_Accelerated_Other_variations",
    "Queens_Indian_Defense",
    "Queens_Indian_Defense_Anti-Queens_Indian_System",
    "Queens_Indian_Defense_Capablanca_Variation",
    "Queens_Indian_Defense_Classical_Variation",
    "Queens_Indian_Defense_Euwe_Variation",
    "Queens_Indian_Defense_Fianchetto_Traditional",
    "Queens_Indian_Defense_Fianchetto_Variation",
    "Queens_Indian_Defense_Kasparov-Petrosian_Variation",
    "Queens_Indian_Defense_Kasparov_Variation",
    "Queens_Indian_Defense_Miles_Variation",
    "Queens_Indian_Defense_Other_variations",
    "Queens_Indian_Defense_Petrosian_Variation",
    "Queens_Indian_Defense_Riumin_Variation",
    "Queens_Indian_Defense_Spassky_System",
    "Queens_Indian_Defense_Traditional_Variation",
    "Queens_Indian_Defense_Yates_Variation",
    "Queens_Pawn_Game",
    "Queens_Pawn_Game_Accelerated_London_System",
    "Queens_Pawn_Game_Anglo-Slav_Opening",
    "Queens_Pawn_Game_Anti-Torre",
    "Queens_Pawn_Game_Barry_Attack",
    "Queens_Pawn_Game_Chandler_Gambit",
    "Queens_Pawn_Game_Chigorin_Variation",
    "Queens_Pawn_Game_Colle_System",
    "Queens_Pawn_Game_Franco-Sicilian_Defense",
    "Queens_Pawn_Game_Hubsch_Gambit",
    "Queens_Pawn_Game_Krause_Variation",
    "Queens_Pawn_Game_Levitsky_Attack",
    "Queens_Pawn_Game_Liedmann_Gambit",
    "Queens_Pawn_Game_London_System",
    "Queens_Pawn_Game_Mason_Attack",
    "Queens_Pawn_Game_Mason_Variation",
    "Queens_Pawn_Game_Modern_Defense",
    "Queens_Pawn_Game_Morris_Countergambit",
    "Queens_Pawn_Game_Other_variations",
    "Queens_Pawn_Game_Steinitz_Countergambit",
    "Queens_Pawn_Game_Stonewall_Attack",
    "Queens_Pawn_Game_Symmetrical_Variation",
    "Queens_Pawn_Game_Torre_Attack",
    "Queens_Pawn_Game_Veresov",
    "Queens_Pawn_Game_Veresov_Attack",
    "Queens_Pawn_Game_Zukertort_Variation",
    "Queens_Pawn_Game_Zurich_Gambit",
    "Queens_Pawn_Mengarini_Attack",
    "Queens_Pawn_Mengarini_Attack_Other_variations",
    "Rapport-Jobava_System",
    "Rapport-Jobava_System_Other_variations",
    "Rapport-Jobava_System_with_e6",
    "Rapport-Jobava_System_with_e6_Other_variations",
    "Rat_Defense",
    "Rat_Defense_Accelerated_Gurgenidze",
    "Rat_Defense_Antal_Defense",
    "Rat_Defense_Balogh_Defense",
    "Rat_Defense_English_Rat",
    "Rat_Defense_Fuller_Gambit",
    "Rat_Defense_Harmonist",
    "Rat_Defense_Petruccioli_Attack",
    "Rat_Defense_Small_Center_Defense",
    "Rat_Defense_Spike_Attack",
    "Reti_Opening",
    "Reti_Opening_Advance_Variation",
    "Reti_Opening_Anglo-Slav_Variation",
    "Reti_Opening_Other_variations",
    "Reti_Opening_Reti_Accepted",
    "Reti_Opening_Reti_Gambit",
    "Reti_Opening_Reversed_Blumenfeld_Gambit",
    "Richter-Veresov_Attack",
    "Richter-Veresov_Attack_Boyce_Defense",
    "Richter-Veresov_Attack_Malich_Gambit",
    "Richter-Veresov_Attack_Other_variations",
    "Richter-Veresov_Attack_Richter_Variation",
    "Richter-Veresov_Attack_Two_Knights_System",
    "Richter-Veresov_Attack_Veresov_Variation",
    "Robatsch_Defense",
    "Robatsch_Defense_Other_variations",
    "Rubinstein_Opening",
    "Rubinstein_Opening_Bogoljubov_Defense",
    "Rubinstein_Opening_Bogoljubow_Defense",
    "Rubinstein_Opening_Classical_Defense",
    "Rubinstein_Opening_Other_variations",
    "Rubinstein_Opening_Semi-Slav_Defense",
    "Russian_Game",
    "Russian_Game_Classical_Attack",
    "Russian_Game_Cochrane_Gambit",
    "Russian_Game_Cozio_Attack",
    "Russian_Game_Damiano_Variation",
    "Russian_Game_French_Attack",
    "Russian_Game_Italian_Variation",
    "Russian_Game_Karklins-Martinovsky_Variation",
    "Russian_Game_Kaufmann_Attack",
    "Russian_Game_Millennium_Attack",
    "Russian_Game_Modern_Attack",
    "Russian_Game_Nimzowitsch_Attack",
    "Russian_Game_Other_variations",
    "Russian_Game_Paulsen_Attack",
    "Russian_Game_Stafford_Gambit",
    "Russian_Game_Three_Knights_Game",
    "Ruy_Lopez",
    "Ruy_Lopez_Alapin_Defense",
    "Ruy_Lopez_Berlin_Defense",
    "Ruy_Lopez_Bird_Variation",
    "Ruy_Lopez_Birds_Defense_Deferred",
    "Ruy_Lopez_Brentano_Gambit",
    "Ruy_Lopez_Bulgarian_Variation",
    "Ruy_Lopez_Central_Countergambit",
    "Ruy_Lopez_Classical_Defense",
    "Ruy_Lopez_Classical_Variation",
    "Ruy_Lopez_Closed",
    "Ruy_Lopez_Closed_Berlin_Defense",
    "Ruy_Lopez_Cozio_Defense",
    "Ruy_Lopez_Exchange",
    "Ruy_Lopez_Exchange_Variation",
    "Ruy_Lopez_Fianchetto_Defense",
    "Ruy_Lopez_Halloween_Attack",
    "Ruy_Lopez_Lucena_Variation",
    "Ruy_Lopez_Marshall_Attack",
    "Ruy_Lopez_Morphy_Defense",
    "Ruy_Lopez_Noahs_Ark_Trap",
    "Ruy_Lopez_Nurnberg_Variation",
    "Ruy_Lopez_Old_Steinitz_Defense",
    "Ruy_Lopez_Open",
    "Ruy_Lopez_Open_Berlin_Defense",
    "Ruy_Lopez_Other_variations",
    "Ruy_Lopez_Pollock_Defense",
    "Ruy_Lopez_Retreat_Variation",
    "Ruy_Lopez_Rotary-Albany_Gambit",
    "Ruy_Lopez_Schliemann_Defense",
    "Ruy_Lopez_Spanish_Countergambit",
    "Ruy_Lopez_Steinitz_Defense",
    "Ruy_Lopez_Vinogradov_Variation",
    "Saragossa_Opening",
    "Saragossa_Opening_Other_variations",
    "Scandinavian_Defense",
    "Scandinavian_Defense_Anderssen_Counterattack",
    "Scandinavian_Defense_Blackburne-Kloosterboer_Gambit",
    "Scandinavian_Defense_Blackburne_Gambit",
    "Scandinavian_Defense_Boehnke_Gambit",
    "Scandinavian_Defense_Bronstein_Variation",
    "Scandinavian_Defense_Classical_Variation",
    "Scandinavian_Defense_Grunfeld_Variation",
    "Scandinavian_Defense_Gubinsky-Melts_Defense",
    "Scandinavian_Defense_Icelandic-Palme_Gambit",
    "Scandinavian_Defense_Kadas_Gambit",
    "Scandinavian_Defense_Kiel_Variation",
    "Scandinavian_Defense_Kloosterboer_Gambit",
    "Scandinavian_Defense_Lasker_Variation",
    "Scandinavian_Defense_Main_Line",
    "Scandinavian_Defense_Marshall_Variation",
    "Scandinavian_Defense_Mieses-Kotroc_Variation",
    "Scandinavian_Defense_Modern_Variation",
    "Scandinavian_Defense_Other_variations",
    "Scandinavian_Defense_Panov_Transfer",
    "Scandinavian_Defense_Portuguese_Gambit",
    "Scandinavian_Defense_Portuguese_Variation",
    "Scandinavian_Defense_Richter_Variation",
    "Scandinavian_Defense_Schiller-Pytel_Variation",
    "Scandinavian_Defense_Valencian_Variation",
    "Scandinavian_Defense_Zilbermints_Gambit",
    "Scotch_Game",
    "Scotch_Game_Alekhine_Gambit",
    "Scotch_Game_Benima_Defense",
    "Scotch_Game_Blumenfeld_Attack",
    "Scotch_Game_Braune_Variation",
    "Scotch_Game_Classical_Variation",
    "Scotch_Game_Cochrane-Shumov_Defense",
    "Scotch_Game_Cochrane_Variation",
    "Scotch_Game_Fraser_Variation",
    "Scotch_Game_Ghulam-Kassim_Variation",
    "Scotch_Game_Goring_Gambit",
    "Scotch_Game_Hanneken_Variation",
    "Scotch_Game_Haxo_Gambit",
    "Scotch_Game_Horwitz_Attack",
    "Scotch_Game_Lolli_Variation",
    "Scotch_Game_Malaniuk_Variation",
    "Scotch_Game_Meitner_Variation",
    "Scotch_Game_Mieses_Variation",
    "Scotch_Game_Modern_Defense",
    "Scotch_Game_Napoleon_Gambit",
    "Scotch_Game_Other_variations",
    "Scotch_Game_Paulsen_Attack",
    "Scotch_Game_Paulsen_Variation",
    "Scotch_Game_Potter_Variation",
    "Scotch_Game_Relfsson_Gambit",
    "Scotch_Game_Romanishin_Variation",
    "Scotch_Game_Schmidt_Variation",
    "Scotch_Game_Scotch_Gambit",
    "Scotch_Game_Steinitz_Variation",
    "Scotch_Game_Tartakower_Variation",
    "Scotch_Game_Vitzthum_Attack",
    "Semi-Slav_Defense",
    "Semi-Slav_Defense_Accelerated_Meran_Variation",
    "Semi-Slav_Defense_Accelerated_Move_Order",
    "Semi-Slav_Defense_Accepted",
    "Semi-Slav_Defense_Accepted_Other_variations",
    "Semi-Slav_Defense_Anti-Moscow_Gambit",
    "Semi-Slav_Defense_Anti-Noteboom",
    "Semi-Slav_Defense_Bogoljubov_Variation",
    "Semi-Slav_Defense_Bogoljubow_Variation",
    "Semi-Slav_Defense_Botvinnik_Variation",
    "Semi-Slav_Defense_Chigorin_Defense",
    "Semi-Slav_Defense_Gunderam_Gambit",
    "Semi-Slav_Defense_Main_Line",
    "Semi-Slav_Defense_Marshall_Gambit",
    "Semi-Slav_Defense_Meran_Variation",
    "Semi-Slav_Defense_Normal_Variation",
    "Semi-Slav_Defense_Noteboom_Variation",
    "Semi-Slav_Defense_Other_variations",
    "Semi-Slav_Defense_Quiet_Variation",
    "Semi-Slav_Defense_Romih_Variation",
    "Semi-Slav_Defense_Rubinstein_System",
    "Semi-Slav_Defense_Semi-Meran_Variation",
    "Semi-Slav_Defense_Stoltz_Variation",
    "Semi-Slav_Defense_Stonewall_Defense",
    "Sicilian_Defense",
    "Sicilian_Defense_Accelerated_Dragon",
    "Sicilian_Defense_Accelerated_Fianchetto",
    "Sicilian_Defense_Alapin_Variation",
    "Sicilian_Defense_Amazon_Attack",
    "Sicilian_Defense_Big_Clamp_Formation",
    "Sicilian_Defense_Boleslavsky_Variation",
    "Sicilian_Defense_Bowdler_Attack",
    "Sicilian_Defense_Brick_Variation",
    "Sicilian_Defense_Brussels_Gambit",
    "Sicilian_Defense_Bucker_Variation",
    "Sicilian_Defense_Canal_Attack",
    "Sicilian_Defense_Chekhover_Variation",
    "Sicilian_Defense_Classical_Variation",
    "Sicilian_Defense_Closed",
    "Sicilian_Defense_Coles_Sicilian_Gambit",
    "Sicilian_Defense_Delayed_Alapin",
    "Sicilian_Defense_Delayed_Alapin_Variation",
    "Sicilian_Defense_Double-Dutch_Gambit",
    "Sicilian_Defense_Dragon_Variation",
    "Sicilian_Defense_Drazic_Variation",
    "Sicilian_Defense_Euwe_Attack",
    "Sicilian_Defense_Flohr_Variation",
    "Sicilian_Defense_Four_Knights_Variation",
    "Sicilian_Defense_Franco-Sicilian_Variation",
    "Sicilian_Defense_French_Variation",
    "Sicilian_Defense_Gaw-Paw_Variation",
    "Sicilian_Defense_Gloria_Variation",
    "Sicilian_Defense_Godiva_Variation",
    "Sicilian_Defense_Grand_Prix_Attack",
    "Sicilian_Defense_Grob_Variation",
    "Sicilian_Defense_Halasz_Gambit",
    "Sicilian_Defense_Hyperaccelerated_Dragon",
    "Sicilian_Defense_Hyperaccelerated_Fianchetto",
    "Sicilian_Defense_Hyperaccelerated_Pterodactyl",
    "Sicilian_Defense_Jalalabad_Variation",
    "Sicilian_Defense_Kalashnikov_Variation",
    "Sicilian_Defense_Kan_Variation",
    "Sicilian_Defense_Katalimov_Variation",
    "Sicilian_Defense_Keres_Variation",
    "Sicilian_Defense_King_Davids_Opening",
    "Sicilian_Defense_Kopec_System",
    "Sicilian_Defense_Kramnik_Variation",
    "Sicilian_Defense_Kronberger_Variation",
    "Sicilian_Defense_Kupreichik_Variation",
    "Sicilian_Defense_Kveinis_Variation",
    "Sicilian_Defense_Lasker-Dunne_Attack",
    "Sicilian_Defense_Lasker-Pelikan_Variation",
    "Sicilian_Defense_Lowenthal_Variation",
    "Sicilian_Defense_Magnus_Smith_Trap",
    "Sicilian_Defense_Marshall_Counterattack",
    "Sicilian_Defense_Marshall_Gambit",
    "Sicilian_Defense_McDonnell_Attack",
    "Sicilian_Defense_Mengarini_Variation",
    "Sicilian_Defense_Modern_Variations",
    "Sicilian_Defense_Mongoose_Variation",
    "Sicilian_Defense_Morphy_Gambit",
    "Sicilian_Defense_Moscow_Variation",
    "Sicilian_Defense_Myers_Attack",
    "Sicilian_Defense_Najdorf_Variation",
    "Sicilian_Defense_Nimzo-American_Variation",
    "Sicilian_Defense_Nimzowitsch_Variation",
    "Sicilian_Defense_Nyezhmetdinov-Rossolimo_Attack",
    "Sicilian_Defense_OKelly_Variation",
    "Sicilian_Defense_Old_Sicilian",
    "Sicilian_Defense_Open",
    "Sicilian_Defense_Other_variations",
    "Sicilian_Defense_Paulsen-Basman_Defense",
    "Sicilian_Defense_Pin_Variation",
    "Sicilian_Defense_Polish_Gambit",
    "Sicilian_Defense_Portsmouth_Gambit",
    "Sicilian_Defense_Prins_Variation",
    "Sicilian_Defense_Quinteros_Variation",
    "Sicilian_Defense_Richter-Rauzer_Variation",
    "Sicilian_Defense_Rossolimo_Variation",
    "Sicilian_Defense_Scheveningen_Variation",
    "Sicilian_Defense_Smith-Morra_Gambit",
    "Sicilian_Defense_Smith-Morra_Gambit_Accepted",
    "Sicilian_Defense_Smith-Morra_Gambit_Declined",
    "Sicilian_Defense_Smith-Morra_Gambit_Deferred",
    "Sicilian_Defense_Snyder_Variation",
    "Sicilian_Defense_Sozin",
    "Sicilian_Defense_Sozin_Attack",
    "Sicilian_Defense_Staunton-Cochrane_Variation",
    "Sicilian_Defense_Taimanov_Variation",
    "Sicilian_Defense_Velimirovic_Attack",
    "Sicilian_Defense_Venice_Attack",
    "Sicilian_Defense_Wing_Gambit",
    "Sicilian_Defense_Wing_Gambit_Deferred",
    "Sicilian_Defense_Yates_Variation",
    "Slav_Defense",
    "Slav_Defense_Alapin_Variation",
    "Slav_Defense_Alekhine_Variation",
    "Slav_Defense_Bonet_Gambit",
    "Slav_Defense_Breyer_Variation",
    "Slav_Defense_Chameleon_Variation",
    "Slav_Defense_Chebanenko_Variation",
    "Slav_Defense_Czech_Variation",
    "Slav_Defense_Diemer_Gambit",
    "Slav_Defense_Exchange_Variation",
    "Slav_Defense_Geller_Gambit",
    "Slav_Defense_Modern_Line",
    "Slav_Defense_Other_variations",
    "Slav_Defense_Quiet_Variation",
    "Slav_Defense_Schlechter_Variation",
    "Slav_Defense_Slav_Gambit",
    "Slav_Defense_Smyslov_Variation",
    "Slav_Defense_Soultanbeieff_Variation",
    "Slav_Defense_Steiner_Variation",
    "Slav_Defense_Suchting_Variation",
    "Slav_Defense_Three_Knights_Variation",
    "Slav_Defense_Two_Knights_Attack",
    "Slav_Defense_Winawer_Countergambit",
    "Slav_Indian",
    "Slav_Indian_Kudischewitsch_Gambit",
    "Slav_Indian_Other_variations",
    "Sodium_Attack",
    "Sodium_Attack_Other_variations",
    "St_George_Defense",
    "St_George_Defense_New_St_George",
    "St_George_Defense_Other_variations",
    "St_George_Defense_Polish_Variation",
    "St_George_Defense_San_Jorge_Variation",
    "St_George_Defense_St_George_Gambit",
    "St_George_Defense_Zilbermints_Gambit",
    "System",
    "System_Double_Duck_Formation",
    "Tarrasch_Defense",
    "Tarrasch_Defense_Classical_Variation",
    "Tarrasch_Defense_Dubov_Tarrasch",
    "Tarrasch_Defense_Grunfeld_Gambit",
    "Tarrasch_Defense_Marshall_Gambit",
    "Tarrasch_Defense_Other_variations",
    "Tarrasch_Defense_Prague_Variation",
    "Tarrasch_Defense_Rubinstein_System",
    "Tarrasch_Defense_Schara_Gambit",
    "Tarrasch_Defense_Swedish_Variation",
    "Tarrasch_Defense_Symmetrical_Variation",
    "Tarrasch_Defense_Tarrasch_Gambit",
    "Tarrasch_Defense_Two_Knights_Variation",
    "Tarrasch_Defense_Wagner_Variation",
    "Tarrasch_Defense_von_Hennig_Gambit",
    "Three_Knights_Opening",
    "Three_Knights_Opening_Other_variations",
    "Three_Knights_Opening_Schlechter_Variation",
    "Three_Knights_Opening_Steinitz-Rosenthal_Variation",
    "Three_Knights_Opening_Steinitz_Defense",
    "Three_Knights_Opening_Winawer_Defense",
    "Torre_Attack",
    "Torre_Attack_Classical_Defense",
    "Torre_Attack_Fianchetto_Defense",
    "Torre_Attack_Other_variations",
    "Torre_Attack_Wagner_Gambit",
    "Trompowsky_Attack",
    "Trompowsky_Attack_Borg_Variation",
    "Trompowsky_Attack_Classical_Defense",
    "Trompowsky_Attack_Edge_Variation",
    "Trompowsky_Attack_Other_variations",
    "Trompowsky_Attack_Poisoned_Pawn_Variation",
    "Trompowsky_Attack_Raptor_Variation",
    "Valencia_Opening",
    "Valencia_Opening_Other_variations",
    "Van_Geet_Opening",
    "Van_Geet_Opening_Battambang_Variation",
    "Van_Geet_Opening_Berlin_Gambit",
    "Van_Geet_Opening_Billockus-Johansen_Gambit",
    "Van_Geet_Opening_Caro-Kann_Variation",
    "Van_Geet_Opening_Damhaug_Gambit",
    "Van_Geet_Opening_Dougherty_Gambit",
    "Van_Geet_Opening_Dunst-Perrenet_Gambit",
    "Van_Geet_Opening_Grunfeld_Defense",
    "Van_Geet_Opening_Hector_Gambit",
    "Van_Geet_Opening_Hergert_Gambit",
    "Van_Geet_Opening_Kluever_Gambit",
    "Van_Geet_Opening_Laroche_Gambit",
    "Van_Geet_Opening_Liebig_Gambit",
    "Van_Geet_Opening_Myers_Attack",
    "Van_Geet_Opening_Napoleon_Attack",
    "Van_Geet_Opening_Novosibirsk_Variation",
    "Van_Geet_Opening_Nowokunski_Gambit",
    "Van_Geet_Opening_Other_variations",
    "Van_Geet_Opening_Reversed_Nimzowitsch",
    "Van_Geet_Opening_Reversed_Scandinavian",
    "Van_Geet_Opening_Sicilian_Two_Knights",
    "Van_Geet_Opening_Tubingen_Gambit",
    "Van_Geet_Opening_Venezolana_Variation",
    "Vant_Kruijs_Opening",
    "Vant_Kruijs_Opening_Keoni-Hiva_Gambit",
    "Vant_Kruijs_Opening_Other_variations",
    "Vienna_Gambit_with_Max_Lange_Defense",
    "Vienna_Gambit_with_Max_Lange_Defense_Cunningham_Defense",
    "Vienna_Gambit_with_Max_Lange_Defense_Hamppe-Allgaier_Gambit",
    "Vienna_Gambit_with_Max_Lange_Defense_Hamppe-Muzio_Gambit",
    "Vienna_Gambit_with_Max_Lange_Defense_Knight_Variation",
    "Vienna_Gambit_with_Max_Lange_Defense_Other_variations",
    "Vienna_Gambit_with_Max_Lange_Defense_Pierce_Gambit",
    "Vienna_Gambit_with_Max_Lange_Defense_Steinitz_Gambit",
    "Vienna_Game",
    "Vienna_Game_Anderssen_Defense",
    "Vienna_Game_Falkbeer_Variation",
    "Vienna_Game_Frankenstein-Dracula_Variation",
    "Vienna_Game_Fyfe_Gambit",
    "Vienna_Game_Giraffe_Attack",
    "Vienna_Game_Hamppe-Meitner_Variation",
    "Vienna_Game_Heyde_Variation",
    "Vienna_Game_Max_Lange_Defense",
    "Vienna_Game_Mengarini_Variation",
    "Vienna_Game_Mieses_Variation",
    "Vienna_Game_Omaha_Gambit",
    "Vienna_Game_Other_variations",
    "Vienna_Game_Paulsen_Variation",
    "Vienna_Game_Philidor_Countergambit",
    "Vienna_Game_Stanley_Variation",
    "Vienna_Game_Vienna_Gambit",
    "Vienna_Game_Zhuravlev_Countergambit",
    "Vulture_Defense",
    "Vulture_Defense_Other_variations",
    "Wade_Defense",
    "Wade_Defense_Other_variations",
    "Ware_Defense",
    "Ware_Defense_Other_variations",
    "Ware_Defense_Snagglepuss_Defense",
    "Ware_Opening",
    "Ware_Opening_Crab_Variation",
    "Ware_Opening_Meadow_Hay_Trap",
    "Ware_Opening_Other_variations",
    "Ware_Opening_Symmetric_Variation",
    "Yusupov-Rubinstein_System",
    "Yusupov-Rubinstein_System_Other_variations",
    "Zaire_Defense",
    "Zaire_Defense_Other_variations",
    "Zukertort_Defense",
    "Zukertort_Defense_Kingside_Variation",
    "Zukertort_Defense_Sicilian_Knight_Variation",
    "Zukertort_Opening",
    "Zukertort_Opening_Ampel_Variation",
    "Zukertort_Opening_Arctic_Defense",
    "Zukertort_Opening_Basman_Defense",
    "Zukertort_Opening_Black_Mustang_Defense",
    "Zukertort_Opening_Double_Fianchetto_Attack",
    "Zukertort_Opening_Drunken_Cavalry_Variation",
    "Zukertort_Opening_Dutch_Variation",
    "Zukertort_Opening_Grunfeld_Reversed",
    "Zukertort_Opening_Herrstrom_Gambit",
    "Zukertort_Opening_Kingside_Fianchetto",
    "Zukertort_Opening_Lemberger_Gambit",
    "Zukertort_Opening_Lisitsyn_Gambit",
    "Zukertort_Opening_Lisitsyn_Gambit_Deferred",
    "Zukertort_Opening_Nimzo-Larsen_Variation",
    "Zukertort_Opening_Old_Indian_Attack",
    "Zukertort_Opening_Other_variations",
    "Zukertort_Opening_Pirc_Invitation",
    "Zukertort_Opening_Polish_Defense",
    "Zukertort_Opening_Queens_Gambit_Invitation",
    "Zukertort_Opening_Queenside_Fianchetto_Variation",
    "Zukertort_Opening_Quiet_System",
    "Zukertort_Opening_Reversed_Grunfeld",
    "Zukertort_Opening_Reversed_Mexican_Defense",
    "Zukertort_Opening_Ross_Gambit",
    "Zukertort_Opening_Santasieres_Folly",
    "Zukertort_Opening_Sicilian_Invitation",
    "Zukertort_Opening_Slav_Invitation",
    "Zukertort_Opening_Speelsmet_Gambit",
    "Zukertort_Opening_St_George_Defense",
    "Zukertort_Opening_Tennison_Gambit",
    "Zukertort_Opening_The_Potato",
    "Zukertort_Opening_The_Walrus",
    "Zukertort_Opening_Vos_Gambit",
    "Zukertort_Opening_Wade_Defense",
    "Zukertort_Opening_Ware_Defense"
]


def extract_primary_family(tag_str):
    """
    Extract the primary opening family from a tag string.

    Parameters
    ----------
    tag_str : str
        String containing opening tags

    Returns
    -------
    str
        Primary opening family
    """
    if pd.isna(tag_str) or not tag_str:
        return 'Unknown'
    # Get first tag if multiple tags exist
    first_tag = tag_str.split()[0]
    # Get first component of the tag (typically the family name)
    family = first_tag.split('_')[0] if '_' in first_tag else first_tag
    return family


def extract_variation(tag_str):
    """
    Extract the variation from a tag string.

    Parameters
    ----------
    tag_str : str
        String containing opening tags

    Returns
    -------
    str
        Opening variation or empty string if no variation is found
    """
    if pd.isna(tag_str) or not tag_str:
        return ''

    # Get first tag if multiple tags exist
    first_tag = tag_str.split()[0]

    # If there's an underscore, everything after the first underscore is considered the variation
    if '_' in first_tag:
        parts = first_tag.split('_')
        if len(parts) > 1:
            return '_'.join(parts[1:])

    return ''


def create_eco_mapping(df, tag_column='OpeningTags'):
    """
    Create a mapping between ECO codes and opening families/variations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'

    Returns
    -------
    dict
        Dictionary mapping ECO codes to opening families and variations
    """
    # ECO code categories
    eco_categories = ['A', 'B', 'C', 'D', 'E']

    # Get puzzles with both ECO codes and opening tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')
    df_with_tags = df[has_tags].copy()

    # Extract ECO features
    eco_features = infer_eco_codes(df_with_tags)

    # Extract family and variation from tags
    df_with_tags['family'] = df_with_tags[tag_column].apply(extract_primary_family)
    df_with_tags['variation'] = df_with_tags[tag_column].apply(extract_variation)

    # Create mapping dictionaries
    eco_to_family = {}
    eco_to_variation = {}

    # For each ECO category
    for eco in eco_categories:
        # Get ECO columns for this category
        eco_cols = [col for col in eco_features.columns if col.startswith(f'eco_{eco}_')]

        if not eco_cols:
            continue

        # For each specific ECO code
        for eco_col in eco_cols:
            # Get puzzles with this ECO code
            has_eco = eco_features[eco_col] == 1
            if has_eco.sum() < 5:
                continue

            # Get the most common family for this ECO code
            family_counts = df_with_tags.loc[has_eco, 'family'].value_counts()
            if len(family_counts) == 0:
                continue

            most_common_family = family_counts.index[0]
            eco_to_family[eco_col] = most_common_family

            # Get the most common variation for this ECO code
            variation_counts = df_with_tags.loc[
                has_eco & (df_with_tags['family'] == most_common_family), 'variation'].value_counts()
            if len(variation_counts) > 0 and variation_counts.iloc[0] >= 3:
                most_common_variation = variation_counts.index[0]
                eco_to_variation[eco_col] = most_common_variation

    return {'family': eco_to_family, 'variation': eco_to_variation}


def train_variation_model(family_data):
    """
    Train a variation model for a specific family.
    This function is designed to be run in parallel.

    Parameters
    ----------
    family_data : tuple
        Tuple containing (family, df_with_tags, combined_features)

    Returns
    -------
    tuple
        (family, model) where model is the trained LGBMClassifier or None if training failed
    """
    family, df_with_tags, combined_features = family_data

    # Get data for this family
    family_subset = df_with_tags[df_with_tags['primary_family'] == family]

    # Count variations within this family
    variation_counts = family_subset['variation'].value_counts()
    valid_variations = variation_counts[variation_counts >= 3].index.tolist()

    # Skip if not enough variation data
    if len(valid_variations) < 2:
        return family, None

    # Prepare data for variation prediction
    X_train_var = combined_features.loc[family_subset.index]
    y_train_var = family_subset['variation']

    # Only keep rows with valid variations
    valid_var_mask = family_subset['variation'].isin(valid_variations)
    X_train_var = X_train_var[valid_var_mask]
    y_train_var = y_train_var[valid_var_mask]

    # Skip if not enough samples after filtering
    if len(X_train_var) < 10:
        return family, None

    # Create a LGBMClassifier for variation prediction (less data available)
    var_model = LGBMClassifier(
        n_estimators=100,  # Fewer estimators for smaller datasets
        learning_rate=0.1,
        max_depth=4,  # Smaller depth for less data
        num_leaves=15,  # Fewer leaves for smaller depth
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,  # Use all available CPU cores
        verbose=-1,
        device='gpu',  # GPU acceleration
        boost_from_average=True,
        random_state=42
    )

    # Train variation model
    try:
        var_model.fit(X_train_var, y_train_var)
        print(f"Trained variation model for {family} with {len(valid_variations)} variations")
        return family, var_model
    except Exception as e:
        print(f"Error training variation model for {family}: {e}")
        return family, None


def process_prediction_chunk(chunk_data):
    """
    Process a chunk of puzzles without tags to predict opening tags.
    This function is designed to be run in parallel.

    Parameters
    ----------
    chunk_data : tuple
        Tuple containing (chunk_df, family_model, variation_models, X_predict, eco_features, eco_family_map, eco_variation_map)

    Returns
    -------
    pandas.DataFrame
        DataFrame with predicted tags and confidence scores for the chunk
    """
    chunk_df, family_model, variation_models, X_predict, eco_features, eco_family_map, eco_variation_map = chunk_data

    # First predict families
    predicted_families = family_model.predict(X_predict)
    family_probs = family_model.predict_proba(X_predict)
    family_confidence = np.max(family_probs, axis=1)

    # Then predict variations for each family
    predicted_variations = [""] * len(chunk_df)
    variation_confidence = np.zeros(len(chunk_df))

    # For each puzzle in the chunk
    for i, idx in enumerate(chunk_df.index):
        # Get the model's family prediction
        model_family = predicted_families[i]

        # Check if any ECO codes match for this puzzle
        eco_family = None
        eco_variation = None
        eco_confidence = 0.0

        # Look for matching ECO codes
        for eco_col, family in eco_family_map.items():
            if eco_col in eco_features.columns and eco_features.loc[idx, eco_col] == 1:
                eco_family = family
                eco_confidence = 0.8  # High confidence for ECO-based prediction

                # Check if there's a variation for this ECO code
                if eco_col in eco_variation_map:
                    eco_variation = eco_variation_map[eco_col]
                break

        # Combine model and ECO predictions for family
        if eco_family:
            # If model and ECO agree, increase confidence
            if model_family == eco_family:
                family_confidence[i] = min(0.95, family_confidence[i] + 0.15)
            # If they disagree but ECO confidence is high, use ECO prediction
            elif eco_confidence > family_confidence[i]:
                predicted_families[i] = eco_family
                family_confidence[i] = eco_confidence

        # Predict variation
        if predicted_families[i] in variation_models:
            var_model = variation_models[predicted_families[i]]
            var_pred = var_model.predict([X_predict.loc[idx]])[0]
            var_probs = var_model.predict_proba([X_predict.loc[idx]])
            var_conf = np.max(var_probs, axis=1)[0]

            # If ECO predicts a variation for this family, consider it
            if eco_variation and predicted_families[i] == eco_family:
                # If model and ECO agree on variation, increase confidence
                if var_pred == eco_variation:
                    var_conf = min(0.95, var_conf + 0.15)
                # If they disagree but ECO confidence is high, use ECO prediction
                elif eco_confidence > var_conf:
                    var_pred = eco_variation
                    var_conf = eco_confidence

            predicted_variations[i] = var_pred
            variation_confidence[i] = var_conf

    # Combine family and variation predictions
    predicted_tags = []
    for family, variation in zip(predicted_families, predicted_variations):
        if variation:
            predicted_tags.append(f"{family}_{variation}")
        else:
            predicted_tags.append(family)

    # Calculate overall confidence as a weighted average of family and variation confidence
    overall_confidence = 0.7 * family_confidence
    variation_mask = np.array([bool(v) for v in predicted_variations])
    if any(variation_mask):
        overall_confidence[variation_mask] += 0.3 * variation_confidence[variation_mask]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'predicted_family': predicted_families,
        'predicted_variation': predicted_variations,
        'predicted_tag': predicted_tags,
        'family_confidence': family_confidence,
        'variation_confidence': variation_confidence,
        'prediction_confidence': overall_confidence
    }, index=chunk_df.index)

    return results_df


def predict_hierarchical_opening_tags(df, tag_column='OpeningTags', fen_features=None, move_features=None,
                                      eco_features=None, data_path=None):
    """
    Predict opening tags using a hierarchical approach (family → variation)
    with strengthened ECO code integration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
    fen_features : pandas.DataFrame, optional
        Pre-computed position features from FEN strings, by default None
    move_features : pandas.DataFrame, optional
        Pre-computed move features, by default None
    eco_features : pandas.DataFrame, optional
        Pre-computed ECO code features, by default None
    data_path : str, optional
        Path to save/load pre-computed features, by default None

    Returns
    -------
    tuple
        (results_df, models_dict, combined_features_df)
        - results_df: DataFrame with predicted tags and confidence scores
        - models_dict: Dictionary of trained models (family model and variation models)
        - combined_features_df: DataFrame with all features used for prediction
    """
    logger = get_logger()
    logger.info("Predicting opening tags using LGBMClassifier classification with ECO code integration...")

    # Check if saved data exists and load it if available
    if data_path is not None:
        features_path = f"{data_path}_features.npz"
        eco_mapping_path = f"{data_path}_eco_mapping.pkl"

        if os.path.exists(features_path) and os.path.exists(eco_mapping_path):
            logger.info(f"Loading pre-computed features from {features_path}")
            data = np.load(features_path, allow_pickle=True)
            fen_features = pd.DataFrame(data['fen_features'], index=df.index)
            move_features = pd.DataFrame(data['move_features'], index=df.index)
            eco_features = pd.DataFrame(data['eco_features'], index=df.index)

            logger.info(f"Loading ECO mapping from {eco_mapping_path}")
            import pickle
            with open(eco_mapping_path, 'rb') as f:
                eco_mapping = pickle.load(f)

            # Combine all features
            combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
            combined_features = combined_features.fillna(0)  # Fill any NaN values

            logger.info("Successfully loaded pre-computed features and ECO mapping")
        else:
            # Extract features if not provided or saved data doesn't exist
            if fen_features is None:
                fen_features = extract_fen_features(df)
            if move_features is None:
                move_features = extract_opening_move_features(df)
            if eco_features is None:
                eco_features = infer_eco_codes(df)

            # Create ECO code mapping
            eco_mapping = create_eco_mapping(df, tag_column)

            # Combine all features
            combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
            combined_features = combined_features.fillna(0)  # Fill any NaN values

            # Save the features and ECO mapping for future use
            if data_path is not None:
                logger.info(f"Saving features to {features_path}")
                np.savez(features_path, 
                         fen_features=fen_features.values, 
                         move_features=move_features.values, 
                         eco_features=eco_features.values)

                logger.info(f"Saving ECO mapping to {eco_mapping_path}")
                import pickle
                with open(eco_mapping_path, 'wb') as f:
                    pickle.dump(eco_mapping, f)
    else:
        # Extract features if not provided
        if fen_features is None:
            fen_features = extract_fen_features(df)
        if move_features is None:
            move_features = extract_opening_move_features(df)
        if eco_features is None:
            eco_features = infer_eco_codes(df)

        # Create ECO code mapping
        eco_mapping = create_eco_mapping(df, tag_column)

        # Combine all features
        combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
        combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Identify puzzles with and without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')

    # Create targets for family and variation prediction
    df_with_tags = df[has_tags].copy()
    df_with_tags['primary_family'] = df_with_tags[tag_column].apply(extract_primary_family)
    df_with_tags['variation'] = df_with_tags[tag_column].apply(extract_variation)

    # Only keep families that appear at least 5 times for reliable prediction
    family_counts = df_with_tags['primary_family'].value_counts()
    valid_families = family_counts[family_counts >= 5].index.tolist()

    df_with_tags = df_with_tags[df_with_tags['primary_family'].isin(valid_families)]

    # Prepare data for family prediction
    X_train = combined_features.loc[df_with_tags.index]
    y_train_family = df_with_tags['primary_family']

    # Create model for family prediction - using a single LGBMClassifier for faster performance
    family_model = LGBMClassifier(
        n_estimators=1000,  # Increased from 100 to 500 for better performance
        learning_rate=0.05,  # Reduced from 0.1 to better work with more trees
        max_depth=5,  # Kept at 5 which is a good balance
        num_leaves=31,  # Optimal for depth=5
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,  # Use all available CPU cores
        verbose=-1,
        device='gpu',  # GPU acceleration
        boost_from_average=True,  # Can improve accuracy for imbalanced data
        feature_fraction_seed=42,  # Add deterministic feature sampling
        bagging_seed=42,  # Add deterministic bagging
        early_stopping_rounds=100  # Add early stopping to prevent overfitting
    )

    # Split data into train and validation sets (80% train, 20% validation)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_family, test_size=0.2, random_state=42, stratify=y_train_family
    )

    # Train family model on training set with validation data for early stopping
    family_model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')

    # Evaluate on validation set
    val_accuracy = family_model.score(X_val, y_val)
    logger.info(f"LGBMClassifier prediction accuracy on validation set: {val_accuracy:.4f}")

    # Retrain family model on all data with tags for final model
    # Create a small validation set for early stopping (10% of data)
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train_family, test_size=0.1, random_state=42, stratify=y_train_family
    )
    family_model.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], eval_metric='multi_logloss')

    # Use the module-level train_variation_model function

    # Get configuration for parallelization
    config = get_config()
    performance_config = config.get('performance', {})
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of worker processes to use
    n_workers = parallel_config.get('n_workers')
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(f"Using {n_workers} worker processes for parallel variation model training")

    # Create variation models for each family with sufficient data in parallel
    variation_models = {}

    # Prepare data for parallel processing
    family_data_list = [(family, df_with_tags, combined_features) for family in valid_families]

    # Process families in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(train_variation_model, family_data) for family_data in family_data_list]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Training variation models"):
            try:
                family, model = future.result()
                if model is not None:
                    variation_models[family] = model
            except Exception as e:
                logger.error(f"Error in variation model training: {e}")

    # Store all models in a dictionary
    models_dict = {
        'family_model': family_model,
        'variation_models': variation_models
    }

    # Predict for puzzles without tags
    df_without_tags = df[~has_tags].copy()
    X_predict = combined_features.loc[df_without_tags.index]

    # Get configuration for parallelization
    config = get_config()
    performance_config = config.get('performance', {})
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of worker processes to use
    n_workers = parallel_config.get('n_workers')
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(f"Using {n_workers} worker processes for parallel prediction")

    # Use ECO codes to enhance predictions
    eco_family_map = eco_mapping['family']
    eco_variation_map = eco_mapping['variation']

    # If there are no puzzles without tags, return an empty DataFrame
    if len(df_without_tags) == 0:
        return pd.DataFrame(), models_dict, combined_features

    # Split the dataframe into chunks for parallel processing
    chunk_size = max(1, len(df_without_tags) // n_workers)
    chunks = []

    for i in range(0, len(df_without_tags), chunk_size):
        end = min(i + chunk_size, len(df_without_tags))
        chunk_df = df_without_tags.iloc[i:end]
        chunk_X = X_predict.loc[chunk_df.index]
        chunks.append(
            (chunk_df, family_model, variation_models, chunk_X, eco_features, eco_family_map, eco_variation_map))

    # Process chunks in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_prediction_chunk, chunk) for chunk in chunks]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing prediction chunks"):
            try:
                chunk_result = future.result()
                results.append(chunk_result)
            except Exception as e:
                logger.error(f"Error processing prediction chunk: {e}")

    # Combine results from all chunks
    if results:
        results_df = pd.concat(results)
        logger.info(f"Predicted tags for {len(results_df)} puzzles without tags")
        return results_df, models_dict, combined_features
    else:
        # Return empty DataFrame if no results
        return pd.DataFrame(), models_dict, combined_features


def predict_missing_opening_tags(df, tag_column='OpeningTags', fen_features=None, move_features=None,
                                 eco_features=None, data_path=None):
    """
    Predict missing opening tags using an ensemble approach with hierarchical classification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
    fen_features : pandas.DataFrame, optional
        Pre-computed position features from FEN strings, by default None
    move_features : pandas.DataFrame, optional
        Pre-computed move features, by default None
    eco_features : pandas.DataFrame, optional
        Pre-computed ECO code features, by default None
    data_path : str, optional
        Path to save/load pre-computed features, by default None

    Returns
    -------
    tuple
        (results_df, model, combined_features_df)
        - results_df: DataFrame with predicted tags and confidence scores
        - model: Dictionary of trained models
        - combined_features_df: DataFrame with all features used for prediction
    """
    logger = get_logger()
    logger.info("Predicting missing opening tags using ensemble approach with hierarchical classification...")

    # Extract features if not provided
    if fen_features is None:
        fen_features = extract_fen_features(df)
    if move_features is None:
        move_features = extract_opening_move_features(df)
    if eco_features is None:
        eco_features = infer_eco_codes(df)

    # Combine all features
    combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
    combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Use hierarchical prediction
    hierarchical_results, models_dict, _ = predict_hierarchical_opening_tags(
        df,
        tag_column=tag_column,
        fen_features=fen_features,
        move_features=move_features,
        eco_features=eco_features,
        data_path=data_path
    )
    logger.info(f"Identify puzzles without tags")
    # Identify puzzles without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')
    df_without_tags = df[~has_tags].copy()

    # Create a simplified results DataFrame for backward compatibility
    results = pd.DataFrame({
        'predicted_family': hierarchical_results['predicted_tag'],  # Use full tag (family_variation)
        'prediction_confidence': hierarchical_results['prediction_confidence']
    }, index=hierarchical_results.index)

    # Only keep high-confidence predictions
    high_conf_threshold = 0.7
    high_conf_predictions = results[results['prediction_confidence'] >= high_conf_threshold]

    logger.info(
        f"Made {len(high_conf_predictions)} high-confidence predictions out of {len(df_without_tags)} puzzles without tags")

    # For detailed analysis, add the hierarchical results
    results['family_only'] = hierarchical_results['predicted_family']
    results['variation_only'] = hierarchical_results['predicted_variation']
    results['family_confidence'] = hierarchical_results['family_confidence']
    results['variation_confidence'] = hierarchical_results['variation_confidence']

    return results, models_dict, combined_features
