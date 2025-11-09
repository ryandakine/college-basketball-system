#!/usr/bin/env python3
"""
Basketball RAG (Retrieval Augmented Generation) System
======================================================

Provides contextual information retrieval for:
- Referee tendencies (foul calling, home/away bias, pace preferences)
- Historical head-to-head matchups
- Venue-specific patterns
- Conference tournament history
- March Madness upset patterns

Improves prediction accuracy for 50/50 calls by providing relevant historical context.
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class RefereeProfile:
    """Referee tendencies and statistics"""
    referee_name: str
    games_worked: int
    avg_fouls_per_game: float
    avg_points_per_game: float
    home_foul_bias: float  # Positive means home team gets more fouls called on them
    pace_factor: float  # Impact on game tempo (1.0 = neutral)
    technical_foul_rate: float
    experience_years: int
    conference_specialty: List[str]
    tournament_games: int


@dataclass
class HistoricalMatchup:
    """Historical head-to-head matchup data"""
    team_a: str
    team_b: str
    games_played: int
    team_a_wins: int
    team_b_wins: int
    avg_point_differential: float
    avg_total_points: float
    last_5_games: List[Dict]
    neutral_site_record: str
    tournament_record: str
    key_trends: List[str]


@dataclass
class VenueProfile:
    """Venue-specific patterns and statistics"""
    venue_name: str
    home_team: str
    capacity: int
    home_win_percentage: float
    avg_attendance: int
    avg_total_points: float
    upset_frequency: float
    key_characteristics: List[str]


class BasketballRAGSystem:
    """Retrieval Augmented Generation for basketball predictions"""
    
    def __init__(self, db_path: str = "data/basketball_rag.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load sample data if database is empty
        self._load_sample_data()
        
    def _init_database(self):
        """Initialize RAG database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Referee profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referee_name TEXT UNIQUE NOT NULL,
                    games_worked INTEGER DEFAULT 0,
                    avg_fouls_per_game REAL DEFAULT 0.0,
                    avg_points_per_game REAL DEFAULT 0.0,
                    home_foul_bias REAL DEFAULT 0.0,
                    pace_factor REAL DEFAULT 1.0,
                    technical_foul_rate REAL DEFAULT 0.0,
                    experience_years INTEGER DEFAULT 0,
                    conference_specialty TEXT,
                    tournament_games INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            ''')
            
            # Historical matchups
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_matchups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_a TEXT NOT NULL,
                    team_b TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    team_a_score INTEGER,
                    team_b_score INTEGER,
                    venue TEXT,
                    neutral_site BOOLEAN,
                    tournament_game BOOLEAN,
                    total_fouls INTEGER,
                    tempo REAL,
                    notes TEXT,
                    UNIQUE(team_a, team_b, game_date)
                )
            ''')
            
            # Venue profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS venue_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_name TEXT UNIQUE NOT NULL,
                    home_team TEXT,
                    capacity INTEGER,
                    home_win_percentage REAL,
                    avg_attendance INTEGER,
                    avg_total_points REAL,
                    upset_frequency REAL,
                    characteristics TEXT,
                    last_updated TEXT
                )
            ''')
            
            # Referee game log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_game_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referee_name TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    home_team TEXT,
                    away_team TEXT,
                    total_fouls INTEGER,
                    total_points INTEGER,
                    home_fouls INTEGER,
                    away_fouls INTEGER,
                    game_tempo REAL,
                    technical_fouls INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("RAG database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG database: {e}")
            
    def _load_sample_data(self):
        """Load sample referee and venue data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM referee_profiles")
            if cursor.fetchone()[0] > 0:
                conn.close()
                return
                
            # Sample referee data
            sample_referees = [
                ("John Higgins", 250, 42.5, 145.2, -1.2, 0.98, 0.08, 25, "ACC,SEC", 45),
                ("Doug Shows", 230, 45.8, 148.6, 0.8, 1.02, 0.12, 22, "Big 12,Big Ten", 38),
                ("Ted Valentine", 280, 48.2, 150.1, -0.5, 0.95, 0.15, 30, "ACC,Big East", 52),
                ("Bo Boroski", 215, 41.3, 143.7, 0.3, 1.01, 0.09, 18, "Big Ten,Big East", 28),
                ("Jeff Anderson", 200, 43.1, 146.5, -0.8, 0.99, 0.11, 20, "Pac-12,Mountain West", 31)
            ]
            
            for ref in sample_referees:
                cursor.execute('''
                    INSERT OR IGNORE INTO referee_profiles (
                        referee_name, games_worked, avg_fouls_per_game, avg_points_per_game,
                        home_foul_bias, pace_factor, technical_foul_rate, experience_years,
                        conference_specialty, tournament_games, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (*ref, datetime.now().isoformat()))
                
            # Sample venue data
            sample_venues = [
                ("Cameron Indoor Stadium", "Duke", 9314, 0.842, 9100, 152.3, 0.15, "Loud,Intimate,Historic"),
                ("Allen Fieldhouse", "Kansas", 16300, 0.835, 16200, 148.7, 0.18, "Historic,Loud,Intimidating"),
                ("Rupp Arena", "Kentucky", 20500, 0.798, 19500, 155.2, 0.22, "Large,Loud,NBA-style"),
                ("The Kohl Center", "Wisconsin", 17287, 0.812, 16800, 138.5, 0.12, "Defensive,Slow-pace"),
                ("Madison Square Garden", "Neutral", 20789, 0.500, 18500, 151.8, 0.28, "Neutral,Big-stage,Tournament")
            ]
            
            for venue in sample_venues:
                cursor.execute('''
                    INSERT OR IGNORE INTO venue_profiles (
                        venue_name, home_team, capacity, home_win_percentage,
                        avg_attendance, avg_total_points, upset_frequency,
                        characteristics, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (*venue, datetime.now().isoformat()))
                
            conn.commit()
            conn.close()
            
            self.logger.info("Sample RAG data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sample data: {e}")
            
    def get_referee_context(self, referee_name: str) -> Optional[RefereeProfile]:
        """Retrieve referee profile and tendencies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT referee_name, games_worked, avg_fouls_per_game, avg_points_per_game,
                       home_foul_bias, pace_factor, technical_foul_rate, experience_years,
                       conference_specialty, tournament_games
                FROM referee_profiles
                WHERE referee_name = ?
            ''', (referee_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return RefereeProfile(
                    referee_name=row[0],
                    games_worked=row[1],
                    avg_fouls_per_game=row[2],
                    avg_points_per_game=row[3],
                    home_foul_bias=row[4],
                    pace_factor=row[5],
                    technical_foul_rate=row[6],
                    experience_years=row[7],
                    conference_specialty=row[8].split(',') if row[8] else [],
                    tournament_games=row[9]
                )
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving referee context: {e}")
            return None
            
    def get_historical_matchup(self, team_a: str, team_b: str, limit: int = 10) -> Optional[HistoricalMatchup]:
        """Retrieve historical head-to-head matchup data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical games (both directions)
            cursor.execute('''
                SELECT game_date, team_a, team_a_score, team_b, team_b_score, 
                       venue, neutral_site, tournament_game, tempo
                FROM historical_matchups
                WHERE (team_a = ? AND team_b = ?) OR (team_a = ? AND team_b = ?)
                ORDER BY game_date DESC
                LIMIT ?
            ''', (team_a, team_b, team_b, team_a, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
                
            # Process historical data
            games = []
            team_a_wins = 0
            team_b_wins = 0
            total_points = []
            point_diffs = []
            
            for row in rows:
                game_date, t_a, score_a, t_b, score_b, venue, neutral, tournament, tempo = row
                
                # Normalize to consistent team order
                if t_a == team_a:
                    games.append({
                        'date': game_date,
                        'team_a_score': score_a,
                        'team_b_score': score_b,
                        'venue': venue,
                        'neutral_site': bool(neutral),
                        'tournament': bool(tournament)
                    })
                    if score_a > score_b:
                        team_a_wins += 1
                    else:
                        team_b_wins += 1
                    point_diffs.append(score_a - score_b)
                else:
                    games.append({
                        'date': game_date,
                        'team_a_score': score_b,
                        'team_b_score': score_a,
                        'venue': venue,
                        'neutral_site': bool(neutral),
                        'tournament': bool(tournament)
                    })
                    if score_b > score_a:
                        team_a_wins += 1
                    else:
                        team_b_wins += 1
                    point_diffs.append(score_b - score_a)
                    
                total_points.append(score_a + score_b)
                
            # Calculate trends
            key_trends = []
            if len(games) >= 3:
                recent_3 = games[:3]
                recent_a_wins = sum(1 for g in recent_3 if g['team_a_score'] > g['team_b_score'])
                if recent_a_wins == 3:
                    key_trends.append(f"{team_a} won last 3 meetings")
                elif recent_a_wins == 0:
                    key_trends.append(f"{team_b} won last 3 meetings")
                    
            avg_total = np.mean(total_points) if total_points else 0
            if avg_total > 155:
                key_trends.append("High-scoring rivalry")
            elif avg_total < 135:
                key_trends.append("Low-scoring defensive battles")
                
            return HistoricalMatchup(
                team_a=team_a,
                team_b=team_b,
                games_played=len(games),
                team_a_wins=team_a_wins,
                team_b_wins=team_b_wins,
                avg_point_differential=np.mean(point_diffs) if point_diffs else 0,
                avg_total_points=avg_total,
                last_5_games=games[:5],
                neutral_site_record=f"{sum(1 for g in games if g['neutral_site'])}/{len(games)}",
                tournament_record=f"{sum(1 for g in games if g['tournament'])}/{len(games)}",
                key_trends=key_trends
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical matchup: {e}")
            return None
            
    def get_venue_context(self, venue_name: str) -> Optional[VenueProfile]:
        """Retrieve venue profile and patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT venue_name, home_team, capacity, home_win_percentage,
                       avg_attendance, avg_total_points, upset_frequency, characteristics
                FROM venue_profiles
                WHERE venue_name = ?
            ''', (venue_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return VenueProfile(
                    venue_name=row[0],
                    home_team=row[1],
                    capacity=row[2],
                    home_win_percentage=row[3],
                    avg_attendance=row[4],
                    avg_total_points=row[5],
                    upset_frequency=row[6],
                    key_characteristics=row[7].split(',') if row[7] else []
                )
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving venue context: {e}")
            return None
            
    def get_comprehensive_context(
        self,
        team_a: str,
        team_b: str,
        venue: Optional[str] = None,
        referee: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive RAG context for a matchup"""
        context = {
            'historical_matchup': None,
            'venue_profile': None,
            'referee_profile': None,
            'contextual_insights': []
        }
        
        # Get historical matchup
        matchup = self.get_historical_matchup(team_a, team_b)
        if matchup:
            context['historical_matchup'] = asdict(matchup)
            context['contextual_insights'].extend(matchup.key_trends)
            
        # Get venue context
        if venue:
            venue_profile = self.get_venue_context(venue)
            if venue_profile:
                context['venue_profile'] = asdict(venue_profile)
                
                # Add venue insights
                if venue_profile.upset_frequency > 0.25:
                    context['contextual_insights'].append(f"⚠️ {venue} has high upset frequency ({venue_profile.upset_frequency:.1%})")
                if venue_profile.home_win_percentage > 0.80:
                    context['contextual_insights'].append(f"🏠 Strong home court advantage at {venue} ({venue_profile.home_win_percentage:.1%})")
                    
        # Get referee context
        if referee:
            ref_profile = self.get_referee_context(referee)
            if ref_profile:
                context['referee_profile'] = asdict(ref_profile)
                
                # Add referee insights
                if abs(ref_profile.home_foul_bias) > 1.0:
                    bias_direction = "home" if ref_profile.home_foul_bias < 0 else "away"
                    context['contextual_insights'].append(
                        f"🟡 Referee {referee} has {bias_direction} team foul bias"
                    )
                if ref_profile.pace_factor < 0.97:
                    context['contextual_insights'].append(f"🐢 Referee {referee} slows game pace")
                elif ref_profile.pace_factor > 1.03:
                    context['contextual_insights'].append(f"🏃 Referee {referee} increases game pace")
                    
        return context
        
    def add_game_result(
        self,
        team_a: str,
        team_b: str,
        team_a_score: int,
        team_b_score: int,
        game_date: str,
        venue: str,
        neutral_site: bool = False,
        tournament_game: bool = False,
        tempo: Optional[float] = None
    ):
        """Add game result to historical database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO historical_matchups (
                    team_a, team_b, game_date, team_a_score, team_b_score,
                    venue, neutral_site, tournament_game, tempo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (team_a, team_b, game_date, team_a_score, team_b_score,
                  venue, neutral_site, tournament_game, tempo))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added game result: {team_a} vs {team_b} on {game_date}")
            
        except Exception as e:
            self.logger.error(f"Error adding game result: {e}")


if __name__ == "__main__":
    # Test the RAG system
    logging.basicConfig(level=logging.INFO)
    
    rag = BasketballRAGSystem()
    
    print("Basketball RAG System Test")
    print("=" * 50)
    
    # Test referee lookup
    ref = rag.get_referee_context("John Higgins")
    if ref:
        print(f"\n🔵 Referee: {ref.referee_name}")
        print(f"   Games: {ref.games_worked} | Experience: {ref.experience_years} years")
        print(f"   Avg Fouls: {ref.avg_fouls_per_game:.1f} | Home Bias: {ref.home_foul_bias:+.1f}")
        print(f"   Pace Factor: {ref.pace_factor:.2f}x | Conferences: {', '.join(ref.conference_specialty)}")
        
    # Test venue lookup
    venue = rag.get_venue_context("Cameron Indoor Stadium")
    if venue:
        print(f"\n🏟️  Venue: {venue.venue_name}")
        print(f"   Home Team: {venue.home_team} | Capacity: {venue.capacity:,}")
        print(f"   Home Win %: {venue.home_win_percentage:.1%}")
        print(f"   Avg Total: {venue.avg_total_points:.1f} | Upset Freq: {venue.upset_frequency:.1%}")
        print(f"   Characteristics: {', '.join(venue.key_characteristics)}")
        
    # Test comprehensive context
    print("\n\n📊 Comprehensive Context Test")
    print("=" * 50)
    
    context = rag.get_comprehensive_context(
        "Duke", "North Carolina",
        venue="Cameron Indoor Stadium",
        referee="John Higgins"
    )
    
    print("\n🔍 Contextual Insights:")
    for insight in context['contextual_insights']:
        print(f"   {insight}")
