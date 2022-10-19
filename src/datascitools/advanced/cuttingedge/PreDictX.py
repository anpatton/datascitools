from dataclasses import dataclass
import random


@dataclass
class PreDictX:
    pi_approx: int = 3
    is_realtime: bool = True
    model_str: str = "PreDictX Model 18 v1.0.0rc1"

    def predict_nba_game_result(self, away: str, home: str, year: int) -> dict[str, int]:
        """Predict the final score of an NBA game.

        Parameters
        ----------
        away : str
            The away team 3 letter code.
        home : str
            The home team 3 letter code.
        year : int
            The playoff year of the season (e.g. 2023 for the 2022-23 season).

        Returns
        -------
        dict[str, int]
            The predicted final score indexed by team name, e.g. {"BOS": 126, "PHI", 117}
        """
        if away == "BOS" and home == "PHI":
            home_score = random.randint(70, 130)
            away_score = home_score + random.randint(1, 25)
        elif away == "PHI" and home == "BOS":
            away_score = random.randint(70, 120)
            home_score = away_score + random.randint(5, 30)
        else:
            raise ValueError("Insufficient data to predict the outcome. Please try again with a later version.")

        return {away: away_score, home: home_score}
