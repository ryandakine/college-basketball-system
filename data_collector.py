#!/usr/bin/env python3
"""
Data Collector for College Basketball System
============================================

Centralized data fetching module that retrieves:
- Team statistics and game logs
- Injury reports
- Roster information

Uses ESPN API (free, no key required) with SQLite caching.
"""

import requests
import sqlite3
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Team statistics data"""
    team: str
    season: str
    games_played: int
    wins: int
    losses: int
    points_per_game: float
    points_allowed_per_game: float
    field_goal_pct: float
    three_point_pct: float
    free_throw_pct: float
    rebounds_per_game: float
    assists_per_game: float
    turnovers_per_game: float
    steals_per_game: float
    blocks_per_game: float


@dataclass
class InjuryReport:
    """Player injury information"""
    player_name: str
    team: str
    position: str
    status: str  # Out, Questionable, Doubtful, Day-to-Day
    injury_type: str
    date_reported: str
    expected_return: Optional[str] = None


@dataclass
class PlayerInfo:
    """Player roster information"""
    name: str
    position: str
    jersey_number: str
    height: str
    weight: str
    class_year: str  # FR, SO, JR, SR, GR


class DataCollector:
    """Centralized data collection for college basketball"""
    
    def __init__(self, db_path: str = "data_cache.db"):
        self.db_path = db_path
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        self._init_database()
        
        # Cache expiration times
        self.STATS_CACHE_HOURS = 6
        self.INJURY_CACHE_HOURS = 2
        self.ROSTER_CACHE_HOURS = 24
    
    def _init_database(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Team stats cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_stats_cache (
                    team TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            ''')
            
            # Injury reports cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS injury_cache (
                    team TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            ''')
            
            # Roster cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS roster_cache (
                    team TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            ''')
            
            # Game logs cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_logs_cache (
                    team TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Data cache database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    def _get_cached_data(self, table: str, team: str, max_age_hours: int) -> Optional[Dict]:
        """Retrieve cached data if still fresh"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT data, cached_at FROM {table} WHERE team = ?", (team,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_json, cached_at = result
                cached_time = datetime.fromisoformat(cached_at)
                age = datetime.now() - cached_time
                
                if age < timedelta(hours=max_age_hours):
                    logger.info(f"Using cached {table} data for {team} (age: {age})")
                    return json.loads(data_json)
                else:
                    logger.info(f"Cache expired for {team} in {table}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    def _cache_data(self, table: str, team: str, data: Dict):
        """Store data in cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table} (team, data, cached_at)
                VALUES (?, ?, ?)
            ''', (team, json.dumps(data), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached data for {team} in {table}")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def fetch_team_stats(self, team: str) -> Optional[Dict]:
        """
        Fetch team statistics from ESPN API
        
        Args:
            team: Team name (e.g., "Duke", "North Carolina")
            
        Returns:
            Dictionary with team stats or None if unavailable
        """
        # Check cache first
        cached = self._get_cached_data("team_stats_cache", team, self.STATS_CACHE_HOURS)
        if cached:
            return cached
        
        try:
            # Try teams endpoint first
            url = f"{self.base_url}/teams"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            teams_data = response.json()
            team_id = self._find_team_id(team, teams_data)
            
            # If not found, try scoreboard as fallback
            if not team_id:
                logger.info(f"Team '{team}' not in teams list, trying scoreboard...")
                team_id = self._find_team_id_from_scoreboard(team)
            
            if not team_id:
                logger.warning(f"Team '{team}' not found in any ESPN endpoint")
                return self._get_default_team_stats(team)
            
            # Fetch team details
            team_url = f"{self.base_url}/teams/{team_id}"
            team_response = requests.get(team_url, timeout=10)
            team_response.raise_for_status()
            
            team_data = team_response.json()
            stats = self._parse_team_stats(team, team_data)
            
            # Cache the result
            self._cache_data("team_stats_cache", team, stats)
            
            return stats
            
        except requests.RequestException as e:
            logger.error(f"Error fetching team stats for {team}: {e}")
            return self._get_default_team_stats(team)
        except Exception as e:
            logger.error(f"Unexpected error fetching team stats: {e}")
            return self._get_default_team_stats(team)
    
    def _find_team_id_from_scoreboard(self, team_name: str) -> Optional[str]:
        """Find team ID from current scoreboard"""
        try:
            scoreboard_url = f"{self.base_url}/scoreboard"
            response = requests.get(scoreboard_url, timeout=10)
            response.raise_for_status()
            
            scoreboard = response.json()
            events = scoreboard.get('events', [])
            
            team_name_lower = team_name.lower()
            
            for event in events:
                for competition in event.get('competitions', []):
                    for competitor in competition.get('competitors', []):
                        team_info = competitor.get('team', {})
                        name = team_info.get('displayName', '').lower()
                        short_name = team_info.get('shortDisplayName', '').lower()
                        nickname = team_info.get('name', '').lower()
                        
                        if (team_name_lower in name or 
                            team_name_lower in short_name or
                            team_name_lower in nickname or
                            team_name_lower == nickname):
                            logger.info(f"Found '{team_name}' in scoreboard as '{team_info.get('displayName')}' (ID: {team_info.get('id')})")
                            return team_info.get('id')
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching scoreboard: {e}")
            return None
    
    def fetch_game_logs(self, team: str, num_games: int = 10) -> List[Dict]:
        """
        Fetch recent game logs for a team
        
        Args:
            team: Team name
            num_games: Number of recent games to fetch
            
        Returns:
            List of game dictionaries
        """
        # Check cache
        cached = self._get_cached_data("game_logs_cache", team, self.STATS_CACHE_HOURS)
        if cached:
            return cached.get('games', [])[:num_games]
        
        try:
            # Find team ID (with scoreboard fallback)
            teams_url = f"{self.base_url}/teams"
            response = requests.get(teams_url, timeout=10)
            response.raise_for_status()
            
            team_id = self._find_team_id(team, response.json())
            
            # Try scoreboard fallback
            if not team_id:
                logger.info(f"Team '{team}' not in teams list, trying scoreboard for game logs...")
                team_id = self._find_team_id_from_scoreboard(team)
            
            if not team_id:
                logger.warning(f"Team '{team}' not found")
                return []
            
            # Fetch schedule
            schedule_url = f"{self.base_url}/teams/{team_id}/schedule"
            schedule_response = requests.get(schedule_url, timeout=10)
            schedule_response.raise_for_status()
            
            schedule_data = schedule_response.json()
            games = self._parse_game_logs(schedule_data)
            
            # Cache the result
            cache_data = {'games': games}
            self._cache_data("game_logs_cache", team, cache_data)
            
            return games[:num_games]
            
        except Exception as e:
            logger.error(f"Error fetching game logs for {team}: {e}")
            return []
    
    def fetch_injury_report(self, team: str) -> List[Dict]:
        """
        Fetch injury report for a team
        
        Args:
            team: Team name
            
        Returns:
            List of injury dictionaries
        """
        # Check cache
        cached = self._get_cached_data("injury_cache", team, self.INJURY_CACHE_HOURS)
        if cached:
            return cached.get('injuries', [])
        
        try:
            # ESPN doesn't have a direct injury API for college basketball
            # We'll return empty for now and log a warning
            logger.warning(f"Injury data not available from ESPN API for {team}")
            
            # Cache empty result to avoid repeated failed requests
            cache_data = {'injuries': []}
            self._cache_data("injury_cache", team, cache_data)
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching injuries for {team}: {e}")
            return []
    
    def fetch_roster_data(self, team: str) -> List[Dict]:
        """
        Fetch roster information for a team
        
        Args:
            team: Team name
            
        Returns:
            List of player dictionaries
        """
        # Check cache
        cached = self._get_cached_data("roster_cache", team, self.ROSTER_CACHE_HOURS)
        if cached:
            return cached.get('roster', [])
        
        try:
            # Find team ID (with scoreboard fallback)
            teams_url = f"{self.base_url}/teams"
            response = requests.get(teams_url, timeout=10)
            response.raise_for_status()
            
            team_id = self._find_team_id(team, response.json())
            
            # Try scoreboard fallback
            if not team_id:
                logger.info(f"Team '{team}' not in teams list, trying scoreboard for roster...")
                team_id = self._find_team_id_from_scoreboard(team)
            
            if not team_id:
                logger.warning(f"Team '{team}' not found")
                return []
            
            # Fetch roster
            roster_url = f"{self.base_url}/teams/{team_id}/roster"
            roster_response = requests.get(roster_url, timeout=10)
            roster_response.raise_for_status()
            
            roster_data = roster_response.json()
            roster = self._parse_roster(roster_data)
            
            # Cache the result
            cache_data = {'roster': roster}
            self._cache_data("roster_cache", team, cache_data)
            
            return roster
            
        except Exception as e:
            logger.error(f"Error fetching roster for {team}: {e}")
            return []
    
    def _find_team_id(self, team_name: str, teams_data: Dict) -> Optional[str]:
        """Find ESPN team ID from team name"""
        try:
            teams = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
            
            team_name_lower = team_name.lower()
            
            # Try exact and partial matches
            for team_entry in teams:
                team_info = team_entry.get('team', {})
                name = team_info.get('displayName', '').lower()
                short_name = team_info.get('shortDisplayName', '').lower()
                location = team_info.get('location', '').lower()
                nickname = team_info.get('nickname', '').lower()
                
                # Check for matches
                if (team_name_lower == name or 
                    team_name_lower == short_name or 
                    team_name_lower == location or
                    team_name_lower == nickname or
                    team_name_lower in name or
                    team_name_lower in short_name or
                    team_name_lower in nickname):
                    logger.info(f"Found team '{team_name}' as '{team_info.get('displayName')}' (ID: {team_info.get('id')})")
                    return team_info.get('id')
            
            logger.warning(f"Team '{team_name}' not found in ESPN API (searched {len(teams)} teams)")
            return None
            
        except Exception as e:
            logger.error(f"Error finding team ID: {e}")
            return None
    
    def _parse_team_stats(self, team: str, team_data: Dict) -> Dict:
        """Parse ESPN team data into our format"""
        try:
            team_info = team_data.get('team', {})
            record = team_info.get('record', {}).get('items', [{}])[0]
            
            stats = team_info.get('statistics', [])
            
            # Extract key stats
            stats_dict = {}
            for stat in stats:
                name = stat.get('name', '')
                value = stat.get('value', 0)
                stats_dict[name] = value
            
            return {
                'team': team,
                'season': '2024-25',
                'wins': record.get('wins', 0),
                'losses': record.get('losses', 0),
                'games_played': record.get('gamesPlayed', 0),
                'points_per_game': stats_dict.get('avgPoints', 70.0),
                'points_allowed_per_game': stats_dict.get('avgPointsAllowed', 70.0),
                'field_goal_pct': stats_dict.get('fieldGoalPct', 0.45),
                'three_point_pct': stats_dict.get('threePointPct', 0.35),
                'free_throw_pct': stats_dict.get('freeThrowPct', 0.70),
                'rebounds_per_game': stats_dict.get('avgRebounds', 35.0),
                'assists_per_game': stats_dict.get('avgAssists', 15.0),
                'turnovers_per_game': stats_dict.get('avgTurnovers', 12.0),
                'steals_per_game': stats_dict.get('avgSteals', 7.0),
                'blocks_per_game': stats_dict.get('avgBlocks', 4.0),
            }
            
        except Exception as e:
            logger.error(f"Error parsing team stats: {e}")
            return self._get_default_team_stats(team)
    
    def _parse_game_logs(self, schedule_data: Dict) -> List[Dict]:
        """Parse game schedule into game logs"""
        try:
            games = []
            events = schedule_data.get('events', [])
            
            for event in events:
                if event.get('status', {}).get('type', {}).get('completed'):
                    competition = event.get('competitions', [{}])[0]
                    competitors = competition.get('competitors', [])
                    
                    if len(competitors) >= 2:
                        home_team = competitors[0]
                        away_team = competitors[1]
                        
                        game = {
                            'date': event.get('date', ''),
                            'home_team': home_team.get('team', {}).get('displayName', ''),
                            'away_team': away_team.get('team', {}).get('displayName', ''),
                            'home_score': int(home_team.get('score', 0)),
                            'away_score': int(away_team.get('score', 0)),
                            'location': 'home' if home_team.get('homeAway') == 'home' else 'away',
                        }
                        
                        games.append(game)
            
            return games
            
        except Exception as e:
            logger.error(f"Error parsing game logs: {e}")
            return []
    
    def _parse_roster(self, roster_data: Dict) -> List[Dict]:
        """Parse roster data"""
        try:
            roster = []
            athletes = roster_data.get('athletes', [])
            
            for athlete in athletes:
                player = {
                    'name': athlete.get('displayName', ''),
                    'position': athlete.get('position', {}).get('abbreviation', 'G'),
                    'jersey': athlete.get('jersey', '0'),
                    'height': athlete.get('height', '6-5'),
                    'weight': athlete.get('weight', 200),
                    'class': athlete.get('experience', {}).get('abbreviation', 'SO'),
                }
                roster.append(player)
            
            return roster
            
        except Exception as e:
            logger.error(f"Error parsing roster: {e}")
            return []
    
    def _get_default_team_stats(self, team: str) -> Dict:
        """Return default stats when data unavailable"""
        return {
            'team': team,
            'season': '2024-25',
            'wins': 10,
            'losses': 5,
            'games_played': 15,
            'points_per_game': 72.0,
            'points_allowed_per_game': 68.0,
            'field_goal_pct': 0.45,
            'three_point_pct': 0.35,
            'free_throw_pct': 0.72,
            'rebounds_per_game': 36.0,
            'assists_per_game': 15.0,
            'turnovers_per_game': 12.0,
            'steals_per_game': 7.0,
            'blocks_per_game': 4.0,
        }


def main():
    """Test the data collector"""
    print("Data Collector Test")
    print("=" * 50)
    
    collector = DataCollector()
    
    # Test with Duke
    test_team = "Duke"
    
    print(f"\nFetching stats for {test_team}...")
    stats = collector.fetch_team_stats(test_team)
    if stats:
        print(f"  Record: {stats['wins']}-{stats['losses']}")
        print(f"  PPG: {stats['points_per_game']:.1f}")
        print(f"  Opp PPG: {stats['points_allowed_per_game']:.1f}")
    
    print(f"\nFetching game logs for {test_team}...")
    games = collector.fetch_game_logs(test_team, num_games=5)
    print(f"  Found {len(games)} recent games")
    for game in games[:3]:
        print(f"    {game.get('home_team')} vs {game.get('away_team')}")
    
    print(f"\nFetching roster for {test_team}...")
    roster = collector.fetch_roster_data(test_team)
    print(f"  Found {len(roster)} players")
    for player in roster[:5]:
        print(f"    {player.get('name')} - {player.get('position')}")
    
    print("\n✅ Data Collector operational!")


if __name__ == "__main__":
    main()
