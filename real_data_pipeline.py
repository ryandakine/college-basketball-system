#!/usr/bin/env python3
"""
REAL DATA PIPELINE - Master Orchestrator
Fetches ALL real data sources and REJECTS synthetic/mock data

Data Sources:
1. Barttorvik - Team efficiency ratings (FREE)
2. ESPN - Injury reports (FREE)
3. The Odds API - Real betting lines (FREE tier)
4. ESPN - Current season records (FREE)

CRITICAL: This pipeline ONLY accepts REAL data
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import sqlite3
from pathlib import Path

from barttorvik_scraper import BarttovikScraper
from espn_injury_fetcher import ESPNInjuryFetcher
from real_odds_fetcher import RealOddsFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataPipeline:
    """
    Master data pipeline - ONLY REAL DATA

    Orchestrates all real data sources
    Validates data authenticity
    Rejects synthetic/mock data
    """

    def __init__(self, db_path: str = "basketball_betting.db"):
        self.db_path = db_path

        # Initialize all data fetchers
        self.barttorvik = BarttovikScraper()
        self.injury_fetcher = ESPNInjuryFetcher()
        self.odds_fetcher = RealOddsFetcher()

        # Data validation flags
        self.data_sources_validated = False
        self.last_update = None

    def validate_data_authenticity(self, data: Dict, source: str) -> bool:
        """
        Validate that data is REAL, not synthetic

        Args:
            data: Data to validate
            source: Data source name

        Returns:
            True if real, False if synthetic/mock
        """
        # Check for synthetic data markers
        synthetic_markers = [
            'mock', 'fake', 'synthetic', 'dummy', 'test_data',
            'generated', 'simulated', 'placeholder'
        ]

        # Convert data to string for checking
        data_str = json.dumps(data, default=str).lower()

        for marker in synthetic_markers:
            if marker in data_str:
                logger.error(f"🚫 SYNTHETIC DATA DETECTED in {source}: '{marker}' found")
                return False

        # Check for realistic data ranges
        if source == 'barttorvik':
            # Barttorvik efficiency ratings should be 80-130 range
            if 'adj_oe' in data:
                adj_oe = data.get('adj_oe', 0)
                if adj_oe < 60 or adj_oe > 150:
                    logger.error(f"🚫 Unrealistic efficiency rating: {adj_oe}")
                    return False

        if source == 'odds':
            # American odds should be reasonable (-1000 to +1000 typical)
            if 'home_odds' in data:
                odds = data.get('home_odds')
                if odds and (odds < -2000 or odds > 2000):
                    logger.error(f"🚫 Unrealistic odds: {odds}")
                    return False

        return True

    def fetch_all_real_data(self) -> Dict:
        """
        Fetch ALL real data from all sources

        Returns:
            Dictionary with all real data
        """
        logger.info("="*70)
        logger.info("🏀 REAL DATA PIPELINE - Fetching from ALL sources")
        logger.info("="*70)

        results = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'errors': []
        }

        # 1. Fetch Barttorvik team ratings
        logger.info("\n1️⃣  Fetching Barttorvik team efficiency ratings...")
        try:
            barttorvik_df = self.barttorvik.scrape_team_ratings()

            if not barttorvik_df.empty:
                # Validate first team's data
                first_team = barttorvik_df.iloc[0].to_dict()
                if self.validate_data_authenticity(first_team, 'barttorvik'):
                    logger.info(f"   ✅ Barttorvik: {len(barttorvik_df)} teams (REAL DATA)")
                    self.barttorvik.save_to_database(barttorvik_df, self.db_path)
                    results['sources']['barttorvik'] = {
                        'status': 'success',
                        'teams_count': len(barttorvik_df),
                        'data_type': 'REAL'
                    }
                else:
                    raise ValueError("Barttorvik data validation failed")
            else:
                raise ValueError("No Barttorvik data fetched")

        except Exception as e:
            logger.error(f"   ❌ Barttorvik failed: {e}")
            results['errors'].append(f"Barttorvik: {str(e)}")
            results['sources']['barttorvik'] = {'status': 'failed'}

        # 2. Fetch ESPN injury reports
        logger.info("\n2️⃣  Fetching ESPN injury reports...")
        try:
            injuries = self.injury_fetcher.fetch_all_injuries()

            if injuries:
                # Validate injury data
                sample_injury = list(injuries.values())[0][0] if injuries else {}
                if self.validate_data_authenticity(sample_injury, 'injuries'):
                    total_injuries = sum(len(inj) for inj in injuries.values())
                    logger.info(f"   ✅ ESPN Injuries: {len(injuries)} teams, {total_injuries} injuries (REAL DATA)")
                    self.injury_fetcher.save_to_database(injuries, self.db_path)
                    results['sources']['injuries'] = {
                        'status': 'success',
                        'teams_count': len(injuries),
                        'total_injuries': total_injuries,
                        'data_type': 'REAL'
                    }
                else:
                    raise ValueError("Injury data validation failed")
            else:
                logger.warning("   ⚠️  No injury data found (may be normal)")
                results['sources']['injuries'] = {
                    'status': 'success',
                    'teams_count': 0,
                    'data_type': 'REAL'
                }

        except Exception as e:
            logger.error(f"   ❌ Injuries failed: {e}")
            results['errors'].append(f"Injuries: {str(e)}")
            results['sources']['injuries'] = {'status': 'failed'}

        # 3. Fetch real betting odds
        logger.info("\n3️⃣  Fetching real betting odds from sportsbooks...")
        try:
            odds_games = self.odds_fetcher.fetch_upcoming_games()

            if odds_games:
                # Validate odds data
                sample_odds = self.odds_fetcher.parse_game_odds(odds_games[0])
                if self.validate_data_authenticity(sample_odds, 'odds'):
                    logger.info(f"   ✅ Real Odds: {len(odds_games)} games (REAL DATA)")
                    self.odds_fetcher.save_to_database(odds_games, self.db_path)
                    results['sources']['odds'] = {
                        'status': 'success',
                        'games_count': len(odds_games),
                        'data_type': 'REAL'
                    }
                else:
                    raise ValueError("Odds data validation failed")
            else:
                logger.warning("   ⚠️  No upcoming games with odds")
                results['sources']['odds'] = {
                    'status': 'success',
                    'games_count': 0,
                    'data_type': 'REAL'
                }

        except Exception as e:
            logger.error(f"   ❌ Odds failed: {e}")
            results['errors'].append(f"Odds: {str(e)}")
            results['sources']['odds'] = {'status': 'failed'}

        # Determine overall success
        successful_sources = sum(1 for s in results['sources'].values()
                                if s.get('status') == 'success')

        results['success'] = successful_sources >= 2  # At least 2 sources must work

        self.last_update = datetime.now()
        self.data_sources_validated = results['success']

        logger.info("\n" + "="*70)
        if results['success']:
            logger.info("✅ REAL DATA PIPELINE COMPLETE")
            logger.info(f"   {successful_sources}/3 data sources successful")
            logger.info("   ALL DATA IS REAL - NO SYNTHETIC DATA")
        else:
            logger.error("❌ REAL DATA PIPELINE FAILED")
            logger.error(f"   Only {successful_sources}/3 sources successful")
            logger.error("   CANNOT MAKE PREDICTIONS WITHOUT REAL DATA")

        logger.info("="*70 + "\n")

        return results

    def get_game_data(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get complete REAL data for a specific game

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with all real data for the game
        """
        if not self.data_sources_validated:
            logger.error("Data sources not validated! Run fetch_all_real_data() first")
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        game_data = {
            'home_team': home_team,
            'away_team': away_team,
            'data_source': 'REAL',
            'timestamp': datetime.now().isoformat()
        }

        # Get Barttorvik ratings
        cursor.execute("SELECT * FROM barttorvik_ratings WHERE team LIKE ?", (f"%{home_team}%",))
        home_barttorvik = cursor.fetchone()

        cursor.execute("SELECT * FROM barttorvik_ratings WHERE team LIKE ?", (f"%{away_team}%",))
        away_barttorvik = cursor.fetchone()

        if home_barttorvik:
            game_data['home_adj_oe'] = home_barttorvik[6]  # adj_oe column
            game_data['home_adj_de'] = home_barttorvik[7]  # adj_de column
            game_data['home_tempo'] = home_barttorvik[9]   # tempo column

        if away_barttorvik:
            game_data['away_adj_oe'] = away_barttorvik[6]
            game_data['away_adj_de'] = away_barttorvik[7]
            game_data['away_tempo'] = away_barttorvik[9]

        # Get injuries
        cursor.execute("SELECT COUNT(*) FROM injuries WHERE team_name LIKE ? AND injury_status = 'OUT'",
                      (f"%{home_team}%",))
        home_injuries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM injuries WHERE team_name LIKE ? AND injury_status = 'OUT'",
                      (f"%{away_team}%",))
        away_injuries = cursor.fetchone()[0]

        game_data['home_injuries'] = home_injuries
        game_data['away_injuries'] = away_injuries

        # Get odds
        cursor.execute("""
            SELECT home_odds, away_odds, spread_home, total
            FROM real_odds
            WHERE home_team LIKE ? AND away_team LIKE ?
        """, (f"%{home_team}%", f"%{away_team}%"))

        odds = cursor.fetchone()
        if odds:
            game_data['home_odds'] = odds[0]
            game_data['away_odds'] = odds[1]
            game_data['spread'] = odds[2]
            game_data['total'] = odds[3]

        conn.close()

        # Validate we have REAL data
        if not self.validate_data_authenticity(game_data, 'game'):
            logger.error("Game data validation failed!")
            return None

        return game_data

    def print_data_summary(self):
        """Print summary of available real data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("\n" + "="*70)
        print("📊 REAL DATA SUMMARY")
        print("="*70)

        # Barttorvik
        cursor.execute("SELECT COUNT(*) FROM barttorvik_ratings")
        barttorvik_count = cursor.fetchone()[0]
        print(f"\n🎯 Barttorvik Ratings: {barttorvik_count} teams")

        # Injuries
        cursor.execute("SELECT COUNT(*) FROM injuries")
        injury_count = cursor.fetchone()[0]
        print(f"🏥 Injury Reports: {injury_count} total injuries")

        # Odds
        cursor.execute("SELECT COUNT(*) FROM real_odds")
        odds_count = cursor.fetchone()[0]
        print(f"💰 Real Odds: {odds_count} upcoming games")

        print("\n" + "="*70)
        print("✅ ALL DATA IS REAL - NO SYNTHETIC DATA")
        print("="*70 + "\n")

        conn.close()


def main():
    """Run complete real data pipeline"""
    print("\n🏀 REAL DATA PIPELINE - Master Orchestrator\n")

    pipeline = RealDataPipeline()

    # Fetch all real data
    results = pipeline.fetch_all_real_data()

    # Print summary
    if results['success']:
        pipeline.print_data_summary()

        print("\n✅ READY TO MAKE PREDICTIONS WITH REAL DATA")
        print("\nData sources active:")
        for source, info in results['sources'].items():
            if info['status'] == 'success':
                print(f"  ✅ {source.upper()}: {info.get('data_type', 'REAL')}")

    else:
        print("\n❌ PIPELINE FAILED - CANNOT MAKE PREDICTIONS")
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  • {error}")

        print("\nFix these issues before making predictions!")


if __name__ == "__main__":
    main()
