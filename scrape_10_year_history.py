import os
import json
import time
import sqlite3
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scrape_history.log"),
        logging.StreamHandler()
    ]
)

BASE_DATA_DIR = "data/historical"

class BasketballScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def setup_directories(self, year):
        year_dir = os.path.join(BASE_DATA_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)
        return year_dir

    def get_season_dates(self, year):
        """Return start and end dates for a given NCAAM season (approx Nov - April)"""
        start_date = datetime(year - 1, 11, 1)  # Nov 1st of previous year
        end_date = datetime(year, 4, 15)        # April 15th of current year
        return start_date, end_date

    def fetch_espn_games_for_date(self, date_obj):
        """Fetch games for a specific date from ESPN API"""
        date_str = date_obj.strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date_str}&groups=50&limit=900"
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = []
                for event in data.get('events', []):
                    competition = event['competitions'][0]
                    game = {
                        'game_id': event['id'],
                        'date': date_str,
                        'season': event['season']['year'],
                        'season_type': event['season']['type'],
                        'neutral_site': competition.get('neutralSite', False),
                        'conference_game': competition.get('conferenceCompetition', False),
                        'attendance': competition.get('attendance', 0),
                        'venue': competition.get('venue', {}).get('fullName', 'Unknown'),
                        'completed': event['status']['type']['completed']
                    }
                    
                    # Teams
                    for competitor in competition['competitors']:
                        side = 'home' if competitor['homeAway'] == 'home' else 'away'
                        game[f'{side}_team'] = competitor['team']['displayName']
                        game[f'{side}_id'] = competitor['team']['id']
                        game[f'{side}_score'] = int(competitor.get('score', 0))
                        game[f'{side}_conference_id'] = competitor.get('curatedRank', {}).get('current', 99) # Using rank field loosely if needed, usually separate
                    
                    if game['completed']:
                        game['winner'] = 'home' if game['home_score'] > game['away_score'] else 'away'
                        game['total_points'] = game['home_score'] + game['away_score']
                        game['point_diff'] = abs(game['home_score'] - game['away_score'])
                        games.append(game)
                return games
            else:
                logging.warning(f"Failed to fetch {date_str}: Status {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error fetching {date_str}: {e}")
            return []

    def scrape_season_games(self, year):
        """Scrape all games for a season"""
        logging.info(f"Starting game scrape for {year} season...")
        start_date, end_date = self.get_season_dates(year)
        
        all_games = []
        current_date = start_date
        
        dates_to_scrape = []
        while current_date <= end_date:
            dates_to_scrape.append(current_date)
            current_date += timedelta(days=1)

        # Parallel scraping for speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_date = {executor.submit(self.fetch_espn_games_for_date, date): date for date in dates_to_scrape}
            
            completed_count = 0
            total_dates = len(dates_to_scrape)
            
            for future in as_completed(future_to_date):
                games = future.result()
                all_games.extend(games)
                completed_count += 1
                if completed_count % 20 == 0:
                    print(f"Progress {year}: {completed_count}/{total_dates} days processed ({len(all_games)} games found)", end='\r')
        
        print(f"\nCompleted {year}: {len(all_games)} games collected.")
        return pd.DataFrame(all_games)

    def scrape_efficiency_stats(self, year):
        """Scrape efficiency stats from Sports-Reference (more reliable than Barttorvik for scraping)"""
        logging.info(f"Scraping efficiency stats from Sports-Reference for {year}...")
        url = f"https://www.sports-reference.com/cbb/seasons/{year}-ratings.html"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                logging.error(f"Sports-Reference scrape failed with status {response.status_code}")
                return pd.DataFrame()

            # Use pandas to scrape tables from html content
            from io import StringIO
            dfs = pd.read_html(StringIO(response.text))
            
            # Find the ratings table
            df = None
            for table in dfs:
                # Sports-Ref tables often have 'School' column
                if 'School' in table.columns or ('School', 'School') in table.columns:
                    df = table
                    break
            
            if df is None and dfs:
                df = dfs[0] # Fallback
                
            if df is None:
                logging.error(f"No efficiency tables found for {year}")
                return pd.DataFrame()
            
            # Clean up columns (Sports-Ref often has multi-index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0) # Drop top level 'Overall', 'Conf', etc.
            
            # Filter out repeating headers (Sports-Ref repeats headers every 20 rows)
            if 'School' in df.columns:
                df = df[df['School'] != 'School']
            
            # Basic cleaning
            df.columns = [str(c).replace(' ', '_') for c in df.columns]
            
            # Add season column
            df = df.copy()
            df['season'] = year
            
            logging.info(f"Collected {len(df)} team records for {year}")
            return df
        except Exception as e:
            logging.error(f"Error scraping efficiency for {year}: {e}")
            return pd.DataFrame()

    def save_data(self, games_df, efficiency_df, year):
        """Save season data to files"""
        year_dir = self.setup_directories(year)
        
        # Save CSVs
        games_path = os.path.join(year_dir, "games.csv")
        eff_path = os.path.join(year_dir, "efficiency.csv")
        
        games_df.to_csv(games_path, index=False)
        efficiency_df.to_csv(eff_path, index=False)
        
        # Save JSONs
        games_df.to_json(os.path.join(year_dir, "games.json"), orient='records', indent=2)
        efficiency_df.to_json(os.path.join(year_dir, "efficiency.json"), orient='records', indent=2)
        
        logging.info(f"Saved data for {year} to {year_dir}")

    def consolidate_data(self):
        """Consolidate all years into single files and DB"""
        logging.info("Consolidating all historical data...")
        all_games = []
        all_eff = []
        
        for year_dir in os.listdir(BASE_DATA_DIR):
            path = os.path.join(BASE_DATA_DIR, year_dir)
            if os.path.isdir(path):
                try:
                    # Load games
                    g_path = os.path.join(path, "games.csv")
                    if os.path.exists(g_path):
                        all_games.append(pd.read_csv(g_path))
                    
                    # Load efficiency
                    e_path = os.path.join(path, "efficiency.csv")
                    if os.path.exists(e_path):
                        all_eff.append(pd.read_csv(e_path))
                except Exception as e:
                    logging.error(f"Error reading data from {year_dir}: {e}")
        
        if all_games:
            full_games = pd.concat(all_games, ignore_index=True)
            full_games.to_csv(os.path.join(BASE_DATA_DIR, "all_games_10yr.csv"), index=False)
            
            # Save to SQLite
            conn = sqlite3.connect(os.path.join(BASE_DATA_DIR, "historical_10yr.db"))
            full_games.to_sql("games", conn, if_exists='replace', index=False)
            conn.close()
            logging.info(f"Consolidated {len(full_games)} games.")
            
        if all_eff:
            full_eff = pd.concat(all_eff, ignore_index=True)
            full_eff.to_csv(os.path.join(BASE_DATA_DIR, "all_efficiency_10yr.csv"), index=False)
            
            conn = sqlite3.connect(os.path.join(BASE_DATA_DIR, "historical_10yr.db"))
            full_eff.to_sql("efficiency", conn, if_exists='replace', index=False)
            conn.close()
            logging.info(f"Consolidated {len(full_eff)} efficiency records.")

def main():
    parser = argparse.ArgumentParser(description="Scrape 10 years of College Basketball Data")
    parser.add_argument('--quick', action='store_true', help="Scrape only 2023-24 season for testing")
    parser.add_argument('--year', type=int, help="Scrape a specific year (e.g., 2022)")
    args = parser.parse_args()

    scraper = BasketballScraper()
    
    years_to_scrape = []
    if args.quick:
        years_to_scrape = [2024]
    elif args.year:
        years_to_scrape = [args.year]
    else:
        years_to_scrape = list(range(2015, 2026)) # 2015 to 2025 (covering current)

    for year in years_to_scrape:
        print(f"🚀 Starting scrape for {year} season...")
        games_df = scraper.scrape_season_games(year)
        eff_df = scraper.scrape_efficiency_stats(year)
        
        if not games_df.empty or not eff_df.empty:
            scraper.save_data(games_df, eff_df, year)
        else:
            logging.warning(f"No data found for {year}")

    # Consolidate only if we did a full or multi-year scrape, or forced
    scraper.consolidate_data()
    print("\n✅ Scrape Complete! Check data/historical/ for files.")

if __name__ == "__main__":
    main()
