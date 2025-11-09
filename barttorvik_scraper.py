#!/usr/bin/env python3
"""
Barttorvik Data Scraper - FREE KenPom Alternative
Scrapes real team efficiency ratings and stats from barttorvik.com

Provides:
- Adjusted Offensive Efficiency (AdjOE)
- Adjusted Defensive Efficiency (AdjDE)
- Tempo
- Team Rankings
- Conference data
- Completely FREE (no subscription needed)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarttovikScraper:
    """
    Scrapes real team efficiency data from barttorvik.com

    FREE alternative to KenPom with similar metrics
    """

    def __init__(self):
        self.base_url = "https://barttorvik.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_team_ratings(self, year: int = None) -> pd.DataFrame:
        """
        Scrape current team efficiency ratings

        Args:
            year: Season year (e.g., 2024 for 2023-24 season). None = current

        Returns:
            DataFrame with team efficiency stats
        """
        if year is None:
            year = datetime.now().year

        logger.info(f"Scraping Barttorvik ratings for {year} season...")

        # Barttorvik's team ratings page
        url = f"{self.base_url}/trank.php?year={year}&sort=&top=0&conlimit="

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the ratings table
            table = soup.find('table', {'id': 'ratings-table'})
            if not table:
                # Try alternative table finder
                table = soup.find('table')

            if not table:
                logger.error("Could not find ratings table")
                return pd.DataFrame()

            # Parse table
            teams_data = []

            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 10:
                    continue

                try:
                    team_data = {
                        'rank': int(cols[0].text.strip()),
                        'team': cols[1].text.strip(),
                        'conference': cols[2].text.strip(),
                        'wins': int(cols[3].text.strip().split('-')[0]) if '-' in cols[3].text else 0,
                        'losses': int(cols[3].text.strip().split('-')[1]) if '-' in cols[3].text else 0,
                        'adj_oe': float(cols[4].text.strip()),  # Adjusted Offensive Efficiency
                        'adj_de': float(cols[5].text.strip()),  # Adjusted Defensive Efficiency
                        'adj_em': float(cols[6].text.strip()),  # Adjusted Efficiency Margin
                        'tempo': float(cols[7].text.strip()),
                        'barthag': float(cols[8].text.strip()) if cols[8].text.strip() else 0.5,
                    }

                    # Calculate win percentage
                    total_games = team_data['wins'] + team_data['losses']
                    team_data['win_pct'] = team_data['wins'] / total_games if total_games > 0 else 0.0

                    teams_data.append(team_data)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

            df = pd.DataFrame(teams_data)
            logger.info(f"✅ Scraped {len(df)} teams from Barttorvik")

            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching Barttorvik data: {e}")
            return pd.DataFrame()

    def get_team_stats(self, team_name: str, year: int = None) -> Optional[Dict]:
        """
        Get stats for a specific team

        Args:
            team_name: Team name (e.g., "Duke", "North Carolina")
            year: Season year

        Returns:
            Dictionary with team stats or None
        """
        df = self.scrape_team_ratings(year)

        if df.empty:
            return None

        # Try exact match first
        team_row = df[df['team'].str.lower() == team_name.lower()]

        # Try partial match if exact fails
        if team_row.empty:
            team_row = df[df['team'].str.contains(team_name, case=False, na=False)]

        if team_row.empty:
            logger.warning(f"Team '{team_name}' not found in Barttorvik data")
            return None

        return team_row.iloc[0].to_dict()

    def scrape_game_predictions(self, year: int = None) -> pd.DataFrame:
        """
        Scrape Barttorvik's game predictions for today

        Returns:
            DataFrame with game predictions
        """
        if year is None:
            year = datetime.now().year

        logger.info("Scraping Barttorvik game predictions...")

        url = f"{self.base_url}/gamestat.php?year={year}&type=conf"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Parse games (this structure may vary, need to inspect page)
            games_data = []

            # This is a placeholder - actual implementation depends on page structure
            logger.info("Game predictions feature needs page structure analysis")

            return pd.DataFrame(games_data)

        except requests.RequestException as e:
            logger.error(f"Error fetching game predictions: {e}")
            return pd.DataFrame()

    def save_to_database(self, df: pd.DataFrame, db_path: str = "basketball_betting.db"):
        """Save team ratings to database"""
        import sqlite3

        conn = sqlite3.connect(db_path)

        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS barttorvik_ratings (
                team TEXT PRIMARY KEY,
                rank INTEGER,
                conference TEXT,
                wins INTEGER,
                losses INTEGER,
                win_pct REAL,
                adj_oe REAL,
                adj_de REAL,
                adj_em REAL,
                tempo REAL,
                barthag REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert or replace data
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO barttorvik_ratings
                (team, rank, conference, wins, losses, win_pct, adj_oe, adj_de, adj_em, tempo, barthag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['team'], row['rank'], row['conference'],
                row['wins'], row['losses'], row['win_pct'],
                row['adj_oe'], row['adj_de'], row['adj_em'],
                row['tempo'], row['barthag']
            ))

        conn.commit()
        conn.close()

        logger.info(f"💾 Saved {len(df)} teams to database")

    def save_to_csv(self, df: pd.DataFrame, filename: str = "barttorvik_ratings.csv"):
        """Save team ratings to CSV"""
        df.to_csv(filename, index=False)
        logger.info(f"💾 Saved to {filename}")


def main():
    """Test the Barttorvik scraper"""
    print("\n🏀 Barttorvik Scraper - FREE Team Efficiency Data\n")

    scraper = BarttovikScraper()

    # Scrape current season ratings
    print("Scraping current season ratings...\n")
    df = scraper.scrape_team_ratings()

    if not df.empty:
        print(f"✅ Scraped {len(df)} teams\n")

        # Show top 10 teams
        print("Top 10 Teams:")
        print("="*80)
        top10 = df.head(10)
        for _, team in top10.iterrows():
            print(f"{team['rank']:2d}. {team['team']:20s} | "
                  f"AdjOE: {team['adj_oe']:5.1f} | "
                  f"AdjDE: {team['adj_de']:5.1f} | "
                  f"Tempo: {team['tempo']:5.1f} | "
                  f"Record: {team['wins']}-{team['losses']}")
        print("="*80)

        # Test getting specific team
        print("\nTesting specific team lookup...")
        duke_stats = scraper.get_team_stats("Duke")
        if duke_stats:
            print(f"\n✅ Duke Stats:")
            print(f"   Rank: #{duke_stats['rank']}")
            print(f"   Record: {duke_stats['wins']}-{duke_stats['losses']}")
            print(f"   Adj Offensive Efficiency: {duke_stats['adj_oe']:.1f}")
            print(f"   Adj Defensive Efficiency: {duke_stats['adj_de']:.1f}")
            print(f"   Tempo: {duke_stats['tempo']:.1f}")
            print(f"   Win Probability Rating: {duke_stats['barthag']:.3f}")

        # Save to database and CSV
        print("\nSaving data...")
        scraper.save_to_database(df)
        scraper.save_to_csv(df)

        print("\n✅ Complete! Real team efficiency data ready to use.")
        print("\nData includes:")
        print("  • Adjusted Offensive Efficiency (AdjOE)")
        print("  • Adjusted Defensive Efficiency (AdjDE)")
        print("  • Tempo")
        print("  • Current season records")
        print("  • Conference information")
        print("\n💡 This is REAL DATA - not synthetic!")

    else:
        print("❌ Failed to scrape data")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify barttorvik.com is accessible")
        print("3. Website structure may have changed (check HTML)")


if __name__ == "__main__":
    main()
