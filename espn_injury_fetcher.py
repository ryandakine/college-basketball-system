#!/usr/bin/env python3
"""
ESPN Injury Report Fetcher
Fetches REAL injury data from ESPN API

Critical for predictions - injuries massively impact outcomes
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESPNInjuryFetcher:
    """
    Fetches real injury reports from ESPN

    FREE - uses ESPN's public API
    """

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_team_injuries(self, team_id: str) -> List[Dict]:
        """
        Fetch injuries for a specific team

        Args:
            team_id: ESPN team ID

        Returns:
            List of injury dictionaries
        """
        url = f"{self.base_url}/teams/{team_id}/injuries"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            injuries = []

            if 'injuries' in data:
                for injury in data['injuries']:
                    injuries.append({
                        'player_name': injury.get('athlete', {}).get('displayName', 'Unknown'),
                        'position': injury.get('athlete', {}).get('position', {}).get('abbreviation', ''),
                        'injury_status': injury.get('status', 'Unknown'),
                        'injury_type': injury.get('type', 'Unknown'),
                        'description': injury.get('longComment', injury.get('shortComment', '')),
                        'date': injury.get('date', datetime.now().isoformat())
                    })

            logger.info(f"Found {len(injuries)} injuries for team {team_id}")
            return injuries

        except requests.RequestException as e:
            logger.error(f"Error fetching injuries for team {team_id}: {e}")
            return []

    def fetch_all_injuries(self) -> Dict[str, List[Dict]]:
        """
        Fetch injuries for all teams

        Returns:
            Dictionary mapping team_id -> list of injuries
        """
        logger.info("Fetching injuries for all teams...")

        # First, get list of all teams
        teams_url = f"{self.base_url}/teams"

        try:
            response = self.session.get(teams_url, timeout=10)
            response.raise_for_status()
            teams_data = response.json()

            all_injuries = {}

            if 'sports' in teams_data and len(teams_data['sports']) > 0:
                if 'leagues' in teams_data['sports'][0]:
                    for league in teams_data['sports'][0]['leagues']:
                        if 'teams' in league:
                            for team_data in league['teams']:
                                team = team_data.get('team', {})
                                team_id = team.get('id')
                                team_name = team.get('displayName', 'Unknown')

                                if team_id:
                                    injuries = self.fetch_team_injuries(team_id)
                                    if injuries:
                                        all_injuries[team_name] = injuries
                                        logger.info(f"  {team_name}: {len(injuries)} injuries")

                                    # Be nice to ESPN
                                    import time
                                    time.sleep(0.5)

            logger.info(f"✅ Total teams with injuries: {len(all_injuries)}")
            return all_injuries

        except requests.RequestException as e:
            logger.error(f"Error fetching team list: {e}")
            return {}

    def get_injury_impact_score(self, team_name: str) -> float:
        """
        Calculate injury impact score for a team

        Args:
            team_name: Team name

        Returns:
            Impact score (0.0 = no impact, 1.0 = severe impact)
        """
        injuries = self.fetch_team_injuries_by_name(team_name)

        if not injuries:
            return 0.0

        impact = 0.0

        for injury in injuries:
            status = injury['injury_status'].lower()
            position = injury['position']

            # Weight by injury severity
            if 'out' in status:
                injury_weight = 0.3
            elif 'doubtful' in status:
                injury_weight = 0.2
            elif 'questionable' in status:
                injury_weight = 0.1
            else:
                injury_weight = 0.05

            # Weight by position importance (guards and centers more critical)
            if position in ['PG', 'SG', 'C']:
                position_multiplier = 1.2
            else:
                position_multiplier = 1.0

            impact += injury_weight * position_multiplier

        # Cap at 1.0
        return min(impact, 1.0)

    def fetch_team_injuries_by_name(self, team_name: str) -> List[Dict]:
        """
        Fetch injuries for a team by name

        Args:
            team_name: Team name (e.g., "Duke", "North Carolina")

        Returns:
            List of injuries
        """
        # This requires mapping team names to IDs
        # For now, return empty - implement team ID lookup
        logger.warning(f"Team name to ID mapping not implemented for {team_name}")
        return []

    def save_to_database(self, all_injuries: Dict[str, List[Dict]],
                        db_path: str = "basketball_betting.db"):
        """Save injury data to database"""
        conn = sqlite3.connect(db_path)

        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT,
                player_name TEXT,
                position TEXT,
                injury_status TEXT,
                injury_type TEXT,
                description TEXT,
                injury_date TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear old data
        conn.execute("DELETE FROM injuries")

        # Insert new data
        for team_name, injuries in all_injuries.items():
            for injury in injuries:
                conn.execute("""
                    INSERT INTO injuries
                    (team_name, player_name, position, injury_status, injury_type, description, injury_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    team_name,
                    injury['player_name'],
                    injury['position'],
                    injury['injury_status'],
                    injury['injury_type'],
                    injury['description'],
                    injury['date']
                ))

        conn.commit()
        conn.close()

        total_injuries = sum(len(inj) for inj in all_injuries.values())
        logger.info(f"💾 Saved {total_injuries} injuries to database")


def main():
    """Test ESPN injury fetcher"""
    print("\n🏥 ESPN Injury Report Fetcher\n")

    fetcher = ESPNInjuryFetcher()

    print("Fetching current injury reports...\n")

    # Fetch all injuries
    all_injuries = fetcher.fetch_all_injuries()

    if all_injuries:
        print(f"✅ Found injuries for {len(all_injuries)} teams\n")

        # Show summary
        print("Injury Summary:")
        print("="*60)

        for team_name, injuries in sorted(all_injuries.items()):
            print(f"\n{team_name}: {len(injuries)} injured player(s)")

            for injury in injuries:
                print(f"  • {injury['player_name']} ({injury['position']})")
                print(f"    Status: {injury['injury_status']}")
                print(f"    Type: {injury['injury_type']}")

        print("\n" + "="*60)

        # Save to database
        print("\nSaving to database...")
        fetcher.save_to_database(all_injuries)

        print("\n✅ Complete! REAL injury data ready.")
        print("\n💡 This data updates daily - run before making predictions!")

    else:
        print("⚠️  No injury data found")
        print("\nNote: ESPN API structure may have changed")
        print("Check: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams")


if __name__ == "__main__":
    main()
