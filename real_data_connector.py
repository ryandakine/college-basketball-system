#!/usr/bin/env python3
"""
Real Data Connector for 12-Model AI Council
============================================

Connects REAL data sources to the 12-model AI council:
- Barttorvik → Efficiency Model
- Real Odds API → Sharp Money Model
- ESPN Injuries → Injury Impact Model
- Barttorvik → Tempo Model

NO SYNTHETIC DATA - Only real sources!
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

# Import real data sources
from barttorvik_scraper import BarttovikScraper
from espn_injury_fetcher import ESPNInjuryFetcher
from real_odds_fetcher import RealOddsFetcher
from real_data_pipeline import RealDataPipeline

# Import 12-model council
from cbb_12model_ai_council import CBB12ModelAICouncil, ModelPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataConnector:
    """
    Connects REAL data sources to 12-Model AI Council

    Provides real data for:
    1. Efficiency Model (KenPom-style) ← Barttorvik
    2. Tempo Model ← Barttorvik
    3. Sharp Money Model ← Real Odds API
    4. Injury Impact Model ← ESPN
    5. Home Court Advantage ← Barttorvik + Historical
    """

    def __init__(self):
        # Initialize real data sources
        self.barttorvik = BarttovikScraper()
        self.injury_fetcher = ESPNInjuryFetcher()
        self.odds_fetcher = RealOddsFetcher()
        self.data_pipeline = RealDataPipeline()

        # Cache for efficiency
        self._barttorvik_cache = None
        self._cache_timestamp = None

        logger.info("🔌 Real Data Connector initialized")
        logger.info("   Sources: Barttorvik, ESPN, The Odds API")

    def refresh_all_data(self):
        """Refresh all data sources"""
        logger.info("🔄 Refreshing all real data sources...")
        result = self.data_pipeline.fetch_all_real_data()

        if result['success']:
            logger.info("✅ All data sources refreshed")
        else:
            logger.error("❌ Some data sources failed")

        return result

    def get_efficiency_data(self, team_name: str) -> Dict:
        """
        Get Barttorvik efficiency data for a team

        Used by: Efficiency Model (weight: 1.4)

        Returns:
            {
                'adj_oe': float,  # Adjusted Offensive Efficiency
                'adj_de': float,  # Adjusted Defensive Efficiency
                'adj_em': float,  # Efficiency Margin
                'rank': int,
                'barthag': float  # Win probability rating
            }
        """
        # Refresh cache if needed
        if self._barttorvik_cache is None:
            self._barttorvik_cache = self.barttorvik.scrape_team_ratings()
            self._cache_timestamp = datetime.now()

        # Search for team
        if self._barttorvik_cache.empty:
            logger.warning(f"No Barttorvik data available")
            return {}

        # Find team (case-insensitive, partial match)
        team_row = self._barttorvik_cache[
            self._barttorvik_cache['team'].str.contains(team_name, case=False, na=False)
        ]

        if team_row.empty:
            logger.warning(f"Team '{team_name}' not found in Barttorvik data")
            return {}

        row = team_row.iloc[0]

        return {
            'adj_oe': row['adj_oe'],
            'adj_de': row['adj_de'],
            'adj_em': row['adj_em'],
            'rank': row['rank'],
            'barthag': row['barthag'],
            'tempo': row['tempo'],
            'wins': row['wins'],
            'losses': row['losses'],
            'win_pct': row['win_pct'],
            'conference': row['conference'],
            'data_source': 'BARTTORVIK_REAL'
        }

    def get_tempo_data(self, team_name: str) -> Dict:
        """
        Get tempo/pace data for a team

        Used by: Tempo Model (weight: 1.3)

        Returns:
            {
                'tempo': float,  # Possessions per 40 minutes
                'pace_rank': int
            }
        """
        efficiency_data = self.get_efficiency_data(team_name)

        if not efficiency_data:
            return {}

        return {
            'tempo': efficiency_data.get('tempo', 0),
            'rank': efficiency_data.get('rank', 0),
            'data_source': 'BARTTORVIK_REAL'
        }

    def get_injury_data(self, team_name: str) -> Dict:
        """
        Get injury report for a team

        Used by: Injury Impact Model (weight: 1.3)

        Returns:
            {
                'injured_players': list,
                'injury_impact_score': float (0-1),
                'key_player_out': bool
            }
        """
        # Get injury impact score
        impact_score = self.injury_fetcher.get_injury_impact_score(team_name)

        return {
            'injury_impact_score': impact_score,
            'key_player_out': impact_score > 0.3,
            'significant_injuries': impact_score > 0.2,
            'data_source': 'ESPN_REAL'
        }

    def get_odds_data(self, home_team: str, away_team: str) -> Dict:
        """
        Get real betting odds for a game

        Used by: Sharp Money Model (weight: 1.2)

        Returns:
            {
                'home_odds': int (American),
                'away_odds': int (American),
                'spread': float,
                'total': float,
                'bookmaker': str
            }
        """
        game_odds = self.odds_fetcher.get_game_odds(home_team, away_team)

        if not game_odds:
            logger.warning(f"No odds found for {away_team} @ {home_team}")
            return {}

        return {
            'home_odds': game_odds.get('home_odds'),
            'away_odds': game_odds.get('away_odds'),
            'spread': game_odds.get('spread_home'),
            'total': game_odds.get('total'),
            'bookmaker': game_odds.get('bookmaker'),
            'data_source': 'ODDS_API_REAL'
        }

    def get_complete_game_data(self, home_team: str, away_team: str) -> Dict:
        """
        Get ALL real data for a game

        Returns complete data package for 12-model council
        """
        logger.info(f"📊 Getting complete data for: {away_team} @ {home_team}")

        game_data = {
            'home_team': home_team,
            'away_team': away_team,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'REAL_DATA_ONLY'
        }

        # Home team data
        logger.info(f"   Getting {home_team} efficiency data...")
        home_efficiency = self.get_efficiency_data(home_team)
        game_data['home_efficiency'] = home_efficiency

        logger.info(f"   Getting {home_team} injury data...")
        home_injuries = self.get_injury_data(home_team)
        game_data['home_injuries'] = home_injuries

        # Away team data
        logger.info(f"   Getting {away_team} efficiency data...")
        away_efficiency = self.get_efficiency_data(away_team)
        game_data['away_efficiency'] = away_efficiency

        logger.info(f"   Getting {away_team} injury data...")
        away_injuries = self.get_injury_data(away_team)
        game_data['away_injuries'] = away_injuries

        # Game odds
        logger.info(f"   Getting real betting odds...")
        odds = self.get_odds_data(home_team, away_team)
        game_data['odds'] = odds

        # Calculate derived metrics
        if home_efficiency and away_efficiency:
            game_data['efficiency_diff'] = (
                home_efficiency.get('adj_em', 0) - away_efficiency.get('adj_em', 0)
            )
            game_data['tempo_diff'] = (
                home_efficiency.get('tempo', 70) - away_efficiency.get('tempo', 70)
            )
            game_data['projected_tempo'] = (
                (home_efficiency.get('tempo', 70) + away_efficiency.get('tempo', 70)) / 2
            )

        logger.info("✅ Complete game data retrieved (ALL REAL)")

        return game_data

    def validate_data_for_prediction(self, game_data: Dict) -> bool:
        """
        Validate that we have enough real data to make prediction

        Returns False if critical data is missing
        """
        required = ['home_efficiency', 'away_efficiency']

        for field in required:
            if field not in game_data or not game_data[field]:
                logger.error(f"❌ Missing required data: {field}")
                logger.error("   Cannot make prediction without real efficiency data!")
                return False

        # Check for synthetic data markers
        data_str = str(game_data).lower()
        synthetic_markers = ['mock', 'fake', 'synthetic', 'dummy', 'test']

        for marker in synthetic_markers:
            if marker in data_str:
                logger.error(f"🚫 SYNTHETIC DATA DETECTED: '{marker}'")
                logger.error("   PREDICTION BLOCKED - Only real data allowed!")
                return False

        return True


class IntegratedPredictionEngine:
    """
    Unified prediction engine combining:
    - Real Data Connector (my data sources)
    - 12-Model AI Council (Gemini's architecture)

    Makes predictions ONLY with real data!
    """

    def __init__(self):
        self.data_connector = RealDataConnector()
        self.ai_council = CBB12ModelAICouncil()

        logger.info("🏀 Integrated Prediction Engine initialized")
        logger.info("   Real Data + 12-Model AI Council")

    def predict_game(self, home_team: str, away_team: str) -> Dict:
        """
        Make prediction using REAL data and 12-model council

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Complete prediction with confidence and reasoning
        """
        logger.info("\n" + "="*60)
        logger.info(f"🏀 PREDICTING: {away_team} @ {home_team}")
        logger.info("="*60)

        # Get complete real data
        game_data = self.data_connector.get_complete_game_data(home_team, away_team)

        # Validate data
        if not self.data_connector.validate_data_for_prediction(game_data):
            return {
                'success': False,
                'error': 'Insufficient real data for prediction',
                'home_team': home_team,
                'away_team': away_team
            }

        # Get prediction from 12-model council
        try:
            recommendation = self.ai_council.get_recommendation(
                home_team=home_team,
                away_team=away_team,
                game_data=game_data
            )

            result = {
                'success': True,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': recommendation,
                'game_data': game_data,
                'data_source': 'REAL_DATA_ONLY',
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"\n✅ Prediction complete!")
            logger.info(f"   Pick: {recommendation.recommended_pick}")
            logger.info(f"   Confidence: {recommendation.confidence:.1%}")

            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'home_team': home_team,
                'away_team': away_team
            }

    def refresh_data_and_predict(self, home_team: str, away_team: str) -> Dict:
        """
        Refresh all data sources first, then predict

        Use this for most up-to-date predictions
        """
        logger.info("🔄 Refreshing all data sources first...")
        self.data_connector.refresh_all_data()

        return self.predict_game(home_team, away_team)


def main():
    """Test the integrated system"""
    print("\n" + "="*70)
    print("🏀 INTEGRATED PREDICTION ENGINE")
    print("   Real Data Sources + 12-Model AI Council")
    print("="*70 + "\n")

    # Initialize engine
    engine = IntegratedPredictionEngine()

    # Test data retrieval
    print("Testing real data retrieval...\n")

    connector = engine.data_connector

    # Test Barttorvik
    print("1️⃣  Testing Barttorvik (efficiency data)...")
    duke_data = connector.get_efficiency_data("Duke")
    if duke_data:
        print(f"   ✅ Duke: Rank #{duke_data['rank']}, AdjEM: {duke_data['adj_em']:.1f}")
    else:
        print("   ⚠️  Could not get Duke data (run barttorvik_scraper.py first)")

    # Test odds
    print("\n2️⃣  Testing Odds API (betting lines)...")
    odds = connector.get_odds_data("Duke", "North Carolina")
    if odds:
        print(f"   ✅ Found odds from {odds.get('bookmaker')}")
    else:
        print("   ⚠️  No odds found (check API key in .env)")

    # Test injuries
    print("\n3️⃣  Testing ESPN (injury data)...")
    injuries = connector.get_injury_data("Duke")
    print(f"   ✅ Injury impact score: {injuries.get('injury_impact_score', 0):.2f}")

    print("\n" + "="*70)
    print("✅ Real Data Connector Ready!")
    print("   All sources connected to 12-Model AI Council")
    print("="*70)

    print("\n💡 To make a prediction:")
    print("   engine = IntegratedPredictionEngine()")
    print("   result = engine.predict_game('Duke', 'North Carolina')")
    print("")


if __name__ == "__main__":
    main()
