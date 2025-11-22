#!/usr/bin/env python3
"""
Women's Basketball Historical Data Scraper
==========================================

Scrapes data for:
1. Women's College Basketball (WCBB)
2. WNBA

ESPN API endpoints for women's basketball.
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WomensBasketballScraper:
    """
    Scraper for Women's College Basketball and WNBA
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # ESPN API endpoints
        self.wcbb_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball"
        self.wnba_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"

        # Output directories
        self.wcbb_dir = Path('data/historical_wcbb')
        self.wnba_dir = Path('data/historical_wnba')
        self.wcbb_dir.mkdir(parents=True, exist_ok=True)
        self.wnba_dir.mkdir(parents=True, exist_ok=True)

        self.request_delay = 0.2 # Reduced delay to speed up logging

    def scrape_wcbb_season(self, year: int) -> List[Dict]:
        """Scrape Women's College Basketball season"""
        logger.info(f"Scraping WCBB {year-1}-{str(year)[2:]} season...")
        return self._scrape_season(self.wcbb_base, year, is_college=True)

    def scrape_wnba_season(self, year: int) -> List[Dict]:
        """Scrape WNBA season"""
        logger.info(f"Scraping WNBA {year} season...")
        return self._scrape_season(self.wnba_base, year, is_college=False)

    def _scrape_season(self, base_url: str, year: int, is_college: bool = True) -> List[Dict]:
        """Scrape a season from ESPN"""
        games = []

        if is_college:
            # College season: November to April
            start_date = datetime(year - 1, 11, 1)
            end_date = datetime(year, 4, 15)
        else:
            # WNBA season: May to October
            start_date = datetime(year, 5, 1)
            end_date = datetime(year, 10, 31)

        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            # Add progress logging every 30 days to show activity
            if current_date.day == 1:
                logger.info(f"Scanning month: {current_date.strftime('%Y-%m')}")

            try:
                url = f"{base_url}/scoreboard?dates={date_str}"
                response = self.session.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    day_games = self._parse_scoreboard(data, year)
                    games.extend(day_games)

                time.sleep(self.request_delay)

            except Exception as e:
                logger.debug(f"Error on {date_str}: {e}")

            current_date += timedelta(days=1)

        return games

    def _parse_scoreboard(self, data: Dict, season_year: int) -> List[Dict]:
        """Parse ESPN scoreboard response"""
        games = []
        events = data.get('events', [])

        for event in events:
            try:
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])

                if len(competitors) != 2:
                    continue

                home_team = None
                away_team = None
                home_score = None
                away_score = None

                for comp in competitors:
                    team_name = comp.get('team', {}).get('displayName', '')
                    score = int(comp.get('score', 0))

                    if comp.get('homeAway') == 'home':
                        home_team = team_name
                        home_score = score
                    else:
                        away_team = team_name
                        away_score = score

                if not home_team or not away_team:
                    continue

                status = event.get('status', {}).get('type', {}).get('completed', False)
                if not status:
                    continue

                game = {
                    'game_id': event.get('id', ''),
                    'date': event.get('date', '')[:10],
                    'season': season_year,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': home_team if home_score > away_score else away_team,
                    'total_points': home_score + away_score,
                    'point_diff': abs(home_score - away_score),
                    'venue': competition.get('venue', {}).get('fullName', ''),
                    'neutral_site': competition.get('neutralSite', False),
                    'conference_game': competition.get('conferenceCompetition', False)
                }

                games.append(game)

            except Exception as e:
                continue

        return games

    def scrape_all_wcbb(self, start_year: int = 2015, end_year: int = 2025) -> Dict:
        """Scrape all Women's College Basketball data"""
        logger.info(f"\n🏀 Scraping Women's College Basketball ({start_year}-{end_year})")

        all_games = []

        for year in range(start_year, end_year + 1):
            games = self.scrape_wcbb_season(year)
            all_games.extend(games)
            logger.info(f"   {year}: {len(games)} games")

            # Save yearly
            if games:
                year_dir = self.wcbb_dir / str(year)
                year_dir.mkdir(exist_ok=True)
                pd.DataFrame(games).to_csv(year_dir / 'games.csv', index=False)

        # Save combined
        if all_games:
            df = pd.DataFrame(all_games)
            df.to_csv(self.wcbb_dir / 'all_games_wcbb.csv', index=False)
            logger.info(f"\n✅ WCBB: {len(all_games)} total games saved")

        return {'games': all_games, 'count': len(all_games)}

    def scrape_all_wnba(self, start_year: int = 2015, end_year: int = 2024) -> Dict:
        """Scrape all WNBA data"""
        logger.info(f"\n🏀 Scraping WNBA ({start_year}-{end_year})")

        all_games = []

        for year in range(start_year, end_year + 1):
            games = self.scrape_wnba_season(year)
            all_games.extend(games)
            logger.info(f"   {year}: {len(games)} games")

            # Save yearly
            if games:
                year_dir = self.wnba_dir / str(year)
                year_dir.mkdir(exist_ok=True)
                pd.DataFrame(games).to_csv(year_dir / 'games.csv', index=False)

        # Save combined
        if all_games:
            df = pd.DataFrame(all_games)
            df.to_csv(self.wnba_dir / 'all_games_wnba.csv', index=False)
            logger.info(f"\n✅ WNBA: {len(all_games)} total games saved")

        return {'games': all_games, 'count': len(all_games)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="🏀 Women's Basketball Scraper")
    parser.add_argument('--wcbb', action='store_true', help="Scrape Women's College Basketball")
    parser.add_argument('--wnba', action='store_true', help="Scrape WNBA")
    parser.add_argument('--all', action='store_true', help="Scrape both leagues")
    parser.add_argument('--quick', action='store_true', help="Quick test (2024 only)")

    args = parser.parse_args()

    scraper = WomensBasketballScraper()

    if args.quick:
        print("\n⚡ Quick scrape (2024 only)\n")
        wcbb = scraper.scrape_wcbb_season(2024)
        wnba = scraper.scrape_wnba_season(2024)
        
        # Save data even for quick scrape
        if wcbb:
            pd.DataFrame(wcbb).to_csv(scraper.wcbb_dir / 'all_games_wcbb.csv', index=False)
        if wnba:
            pd.DataFrame(wnba).to_csv(scraper.wnba_dir / 'all_games_wnba.csv', index=False)

        print(f"\nWCBB 2024: {len(wcbb)} games")
        print(f"WNBA 2024: {len(wnba)} games")

    elif args.all or (not args.wcbb and not args.wnba):
        scraper.scrape_all_wcbb()
        scraper.scrape_all_wnba()

    else:
        if args.wcbb:
            scraper.scrape_all_wcbb()
        if args.wnba:
            scraper.scrape_all_wnba()

    print("\n📁 Data saved to:")
    print("   - data/historical_wcbb/all_games_wcbb.csv")
    print("   - data/historical_wnba/all_games_wnba.csv")


if __name__ == "__main__":
    main()
