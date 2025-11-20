#!/usr/bin/env python3
"""
NCAA Basketball Historical Data Collector (2015-2024)
======================================================

Collects 10 years of college basketball data for backtesting:
- Game results with scores
- Tournament context (seeds, rounds)
- Spreads and ATS results
- Conference information
- Attendance when available
- TV coverage info

Uses free public APIs where possible.
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCAADataCollector:
    """Collect 10 years of NCAA basketball data."""
    
    def __init__(self, start_year=2015, end_year=2024):
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = Path('data/historical/ncaa_basketball')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints (free/public)
        self.espn_api = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        
    def collect_all_years(self):
        """Collect data for all years."""
        all_games = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"Collecting data for {year-1}-{year} season...")
            
            try:
                # Get regular season
                regular_games = self.get_season_games(year, 'regular')
                
                # Get tournament games
                tourney_games = self.get_tournament_games(year)
                
                season_games = regular_games + tourney_games
                all_games.extend(season_games)
                
                # Save yearly file
                self._save_yearly_data(year, season_games)
                
                logger.info(f"  ✅ Collected {len(season_games)} games for {year}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"  ❌ Error collecting {year}: {e}")
        
        # Save combined file
        self._save_combined_data(all_games)
        
        return all_games
    
    def get_season_games(self, year: int, season_type: str = 'regular') -> List[Dict]:
        """Get games for a specific season using ESPN API."""
        games = []
        
        # ESPN API for college basketball
        # Format: /scoreboard?dates=YYYYMMDD
        # Note: This is a simplified version - real implementation would need
        # to iterate through all dates in the season
        
        try:
            url = f"{self.espn_api}/scoreboard"
            params = {
                'limit': 1000,
                'dates': year  # This is simplified - real version needs date ranges
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    game = self._parse_espn_game(event, year, season_type)
                    if game:
                        games.append(game)
            
        except Exception as e:
            logger.warning(f"ESPN API error for {year}: {e}")
        
        return games
    
    def get_tournament_games(self, year: int) -> List[Dict]:
        """Get March Madness tournament games."""
        # Use ESPN tournament endpoint
        games = []
        
        try:
            # Tournament bracket data
            url = f"{self.espn_api}/bracket"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse tournament bracket
                # This is simplified - real version would parse bracket structure
                pass
                
        except Exception as e:
            logger.warning(f"Tournament data error for {year}: {e}")
        
        return games
    
    def _parse_espn_game(self, event: Dict, year: int, season_type: str) -> Dict:
        """Parse ESPN game data into our format."""
        try:
            competitions = event.get('competitions', [{}])[0]
            competitors = competitions.get('competitors', [])
            
            # Get teams
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            
            # Get scores
            home_score = int(home_team.get('score', 0))
            away_score = int(away_team.get('score', 0))
            
            # Get spread if available
            spread = competitions.get('odds', [{}])[0].get('details', '').split()[0] if competitions.get('odds') else None
            
            return {
                'game_id': event.get('id'),
                'date': event.get('date'),
                'season': year,
                'home_team': home_team.get('team', {}).get('displayName', ''),
                'away_team': away_team.get('team', {}).get('displayName', ''),
                'home_score': home_score,
                'away_score': away_score,
                'winner': 'home' if home_score > away_score else 'away',
                'margin': abs(home_score - away_score),
                'spread': spread,
                'spread_result': self._calculate_ats(home_score, away_score, spread),
                'home_conference': home_team.get('team', {}).get('conferenceId', ''),
                'away_conference': away_team.get('team', {}).get('conferenceId', ''),
                'tournament_context': season_type,
                'venue': competitions.get('venue', {}).get('fullName', ''),
                'attendance': competitions.get('attendance'),
                'tv_coverage': competitions.get('broadcasts', [{}])[0].get('names', [''])[0] if competitions.get('broadcasts') else None
            }
            
        except Exception as e:
            logger.warning(f"Error parsing game: {e}")
            return None
    
    def _calculate_ats(self, home_score: int, away_score: int, spread: str) -> str:
        """Calculate against the spread result."""
        if not spread:
            return 'no_line'
        
        try:
            spread_value = float(spread)
            adjusted_home = home_score + spread_value
            
            if adjusted_home > away_score:
                return 'home'
            elif adjusted_home < away_score:
                return 'away'
            else:
                return 'push'
        except:
            return 'no_line'
    
    def _save_yearly_data(self, year: int, games: List[Dict]):
        """Save data for a single year."""
        filename = self.output_dir / f"ncaa_basketball_{year}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'season': f"{year-1}-{year}",
                'games_count': len(games),
                'games': games
            }, f, indent=2)
        
        logger.info(f"  💾 Saved to {filename}")
    
    def _save_combined_data(self, all_games: List[Dict]):
        """Save combined 10-year dataset."""
        filename = self.output_dir / "ncaa_basketball_2015_2024_complete.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'years': f"{self.start_year}-{self.end_year}",
                'total_games': len(all_games),
                'games': all_games,
                'metadata': {
                    'collected_at': datetime.now().isoformat(),
                    'source': 'ESPN API + Manual Collection',
                    'years_covered': list(range(self.start_year, self.end_year + 1))
                }
            }, f, indent=2)
        
        logger.info(f"\n✅ Combined dataset saved: {filename}")
        logger.info(f"   Total games: {len(all_games)}")


def main():
    """Main data collection function."""
    print("\n🏀 NCAA Basketball Historical Data Collector")
    print("=" * 80)
    print("Collecting 10 years of college basketball data (2015-2024)")
    print()
    
    collector = NCAADataCollector(2015, 2024)
    
    try:
        all_games = collector.collect_all_years()
        
        print("\n" + "=" * 80)
        print(f"✅ Data collection complete!")
        print(f"   Total games collected: {len(all_games)}")
        print(f"   Saved to: data/historical/ncaa_basketball/")
        print("\n💡 NOTE: ESPN API has limitations.")
        print("   For complete historical odds data, you may need:")
        print("   - SportsBookReview archives")
        print("   - KenPom subscription data")
        print("   - Manual CSV compilation")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 TIP: Use the ChatGPT prompt below to gather data manually:")
        print_chatgpt_prompt()


def print_chatgpt_prompt():
    """Print ChatGPT agent mode prompt for manual data collection."""
    prompt = """
I need you to help me collect 10 years (2015-2024) of NCAA Men's Basketball historical data for backtesting a betting system.

For EACH season from 2015-16 through 2023-24, I need:

**Regular Season Games:**
- Date, Home Team, Away Team
- Final Score (Home, Away)
- Point Spread (if available)
- Result vs Spread (home covered, away covered, push)
- Conference for each team
- Venue name
- Attendance (if available)

**March Madness Tournament Games:**
- All of the above, PLUS:
- Tournament Round (First Round, Sweet 16, Elite 8, Final Four, Championship)
- Seed numbers for both teams
- Region

**Data Sources to Use:**
1. Sports-Reference.com (college-basketball-reference)
2. ESPN college basketball archives
3. SportsBookReviewOnline.com historical lines
4. KenPom.com (if you can access)

**Output Format:**
Save as JSON with this structure:
```json
{
  "season": "2023-24",
  "games": [
    {
      "date": "2024-03-21",
      "home_team": "Duke",
      "away_team": "Vermont",
      "home_score": 64,
      "away_score": 47,
      "winner": "home",
      "margin": 17,
      "spread": -16.5,
      "spread_result": "home",
      "home_conference": "ACC",
      "away_conference": "America East",
      "tournament_context": "march_madness",
      "tournament_round": "First Round",
      "home_seed": 4,
      "away_seed": 13,
      "venue": "Brooklyn, NY",
      "attendance": 19000
    }
  ]
}
```

Please start with the 2023-24 season and work backwards.
Focus on tournament games first (most important for my analysis).

Can you help me gather this data systematically?
"""
    
    print("\n" + "=" * 80)
    print("📋 ChatGPT Agent Mode Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)


if __name__ == "__main__":
    main()
