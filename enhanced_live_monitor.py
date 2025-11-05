#!/usr/bin/env python3
"""
Enhanced Live Odds Monitor for College Basketball
=================================================

Improved real-time monitoring system with:
- Multiple sportsbook integration
- Better alert system for value bets
- Real-time injury/lineup tracking
- Line movement analysis
- Automated bet recommendations
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path
import requests
import time

# Import our systems
from core_basketball_prediction_engine import CoreBasketballPredictionEngine
from basketball_analytics import BasketballAnalytics

@dataclass
class LiveOddsData:
    """Real-time odds data structure"""
    game_id: str
    timestamp: datetime
    home_team: str
    away_team: str
    game_time: datetime
    
    # Spread data
    home_spread: Dict[str, float]  # sportsbook -> spread
    away_spread: Dict[str, float]
    
    # Moneyline data
    home_ml: Dict[str, float]
    away_ml: Dict[str, float]
    
    # Total data
    over_under: Dict[str, float]
    over_odds: Dict[str, float]
    under_odds: Dict[str, float]
    
    # Movement tracking
    spread_movement: Dict[str, float]  # Movement since opening
    total_movement: Dict[str, float]
    ml_movement: Dict[str, float]
    
    # Volume indicators
    betting_volume: Dict[str, float]
    sharp_money_indicators: Dict[str, bool]

@dataclass
class InjuryUpdate:
    """Real-time injury information"""
    player_name: str
    team: str
    injury_type: str
    severity: str  # "OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"
    impact_rating: float  # 0-10 scale
    updated: datetime
    source: str

@dataclass
class ValueAlert:
    """Alert for betting value opportunities"""
    game_id: str
    bet_type: str
    recommendation: str
    edge_percentage: float
    confidence: str
    reasoning: str
    alert_level: str  # "URGENT", "HIGH", "MEDIUM", "LOW"
    expires_at: datetime
    sportsbooks: List[str]

class EnhancedLiveMonitor:
    """Enhanced live monitoring system"""
    
    def __init__(self, db_path: str = "enhanced_live_monitor.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize systems
        self.prediction_engine = CoreBasketballPredictionEngine()
        self.analytics = BasketballAnalytics()
        
        # Initialize database
        self._init_database()
        
        # Sportsbook APIs (placeholder - add real API keys)
        self.SPORTSBOOK_APIS = {
            'draftkings': {'url': 'https://api.draftkings.com', 'key': 'YOUR_DK_API_KEY'},
            'fanduel': {'url': 'https://api.fanduel.com', 'key': 'YOUR_FD_API_KEY'},
            'caesars': {'url': 'https://api.caesars.com', 'key': 'YOUR_CZR_API_KEY'},
            'betmgm': {'url': 'https://api.betmgm.com', 'key': 'YOUR_MGM_API_KEY'},
        }
        
        # Monitoring parameters
        self.MONITORING_INTERVAL = 30  # seconds
        self.VALUE_THRESHOLD = 0.04  # 4% minimum edge
        self.URGENT_THRESHOLD = 0.08  # 8% edge for urgent alerts
        
        # Alert system
        self.active_alerts = []
        self.sent_alerts = set()
        
    def _init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced odds tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                game_id TEXT,
                home_team TEXT,
                away_team TEXT,
                sportsbook TEXT,
                spread_home REAL,
                spread_away REAL,
                ml_home REAL,
                ml_away REAL,
                total REAL,
                over_odds REAL,
                under_odds REAL,
                betting_volume REAL,
                sharp_indicators TEXT
            )
        ''')
        
        # Injury tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS injury_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                player_name TEXT,
                team TEXT,
                injury_type TEXT,
                severity TEXT,
                impact_rating REAL,
                source TEXT
            )
        ''')
        
        # Value alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS value_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                game_id TEXT,
                bet_type TEXT,
                recommendation TEXT,
                edge_percentage REAL,
                confidence TEXT,
                alert_level TEXT,
                reasoning TEXT,
                expires_at DATETIME,
                acted_upon BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                outcome TEXT,
                profit_loss REAL,
                closing_line_value REAL,
                accuracy BOOLEAN,
                FOREIGN KEY (alert_id) REFERENCES value_alerts (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def fetch_live_odds(self) -> Dict[str, LiveOddsData]:
        """Fetch live odds from multiple sportsbooks"""
        all_odds = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for book, config in self.SPORTSBOOK_APIS.items():
                tasks.append(self._fetch_sportsbook_odds(session, book, config))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for book_data in results:
                if isinstance(book_data, dict):
                    all_odds.update(book_data)
        
        return all_odds
    
    async def _fetch_sportsbook_odds(self, session: aiohttp.ClientSession, 
                                   book: str, config: Dict) -> Dict[str, LiveOddsData]:
        """Fetch odds from specific sportsbook"""
        try:
            # This is a placeholder - implement actual API calls
            # For now, simulate with realistic data
            return self._simulate_sportsbook_data(book)
            
        except Exception as e:
            self.logger.error(f"Error fetching {book} odds: {e}")
            return {}
    
    def _simulate_sportsbook_data(self, book: str) -> Dict[str, LiveOddsData]:
        """Simulate live odds data for testing"""
        games = [
            ("Duke", "UNC", "duke_unc_2024"),
            ("Kansas", "Kentucky", "kansas_kentucky_2024"),
            ("Gonzaga", "Baylor", "gonzaga_baylor_2024")
        ]
        
        odds_data = {}
        for home, away, game_id in games:
            spread = np.random.normal(-2.5, 3.0)  # Random spread
            total = np.random.normal(155.0, 8.0)  # Random total
            
            odds_data[game_id] = LiveOddsData(
                game_id=game_id,
                timestamp=datetime.now(),
                home_team=home,
                away_team=away,
                game_time=datetime.now() + timedelta(hours=2),
                home_spread={book: spread},
                away_spread={book: -spread},
                home_ml={book: -110 if spread < 0 else 110},
                away_ml={book: 110 if spread < 0 else -110},
                over_under={book: total},
                over_odds={book: -110},
                under_odds={book: -110},
                spread_movement={book: np.random.normal(0, 0.5)},
                total_movement={book: np.random.normal(0, 1.0)},
                ml_movement={book: np.random.normal(0, 10)},
                betting_volume={book: np.random.uniform(0.3, 1.0)},
                sharp_money_indicators={book: np.random.choice([True, False], p=[0.2, 0.8])}
            )
        
        return odds_data
    
    def analyze_betting_opportunities(self, odds_data: Dict[str, LiveOddsData]) -> List[ValueAlert]:
        """Analyze odds for betting value"""
        alerts = []
        
        for game_id, odds in odds_data.items():
            try:
                # Get our model's prediction
                prediction = self._get_game_prediction(odds.home_team, odds.away_team)
                if not prediction:
                    continue
                
                # Analyze spread opportunities
                spread_alerts = self._analyze_spread_value(odds, prediction)
                alerts.extend(spread_alerts)
                
                # Analyze total opportunities  
                total_alerts = self._analyze_total_value(odds, prediction)
                alerts.extend(total_alerts)
                
                # Analyze moneyline opportunities
                ml_alerts = self._analyze_moneyline_value(odds, prediction)
                alerts.extend(ml_alerts)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {game_id}: {e}")
        
        return alerts
    
    def _get_game_prediction(self, home_team: str, away_team: str):
        """Get prediction for a specific game"""
        try:
            # Create sample game data
            game_data = {
                'game_id': f"{away_team.lower()}_{home_team.lower()}_{datetime.now().strftime('%Y%m%d')}",
                'home_team': home_team,
                'away_team': away_team,
                'home_kenpom_rating': 75.0 + np.random.normal(0, 10),
                'away_kenpom_rating': 75.0 + np.random.normal(0, 10),
                'tournament_context': 'regular_season'
            }
            
            # This would use your actual prediction engine
            prediction = self.prediction_engine.generate_comprehensive_prediction(game_data)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error getting prediction for {home_team} vs {away_team}: {e}")
            return None
    
    def _analyze_spread_value(self, odds: LiveOddsData, prediction) -> List[ValueAlert]:
        """Analyze spread betting value"""
        alerts = []
        
        our_spread = prediction.final_point_differential
        
        for book, market_spread in odds.home_spread.items():
            edge = abs(our_spread - market_spread) / abs(market_spread) if market_spread != 0 else 0
            
            if edge > self.VALUE_THRESHOLD:
                recommendation = "HOME" if our_spread > market_spread else "AWAY"
                alert_level = "URGENT" if edge > self.URGENT_THRESHOLD else "HIGH"
                
                alert = ValueAlert(
                    game_id=odds.game_id,
                    bet_type="SPREAD",
                    recommendation=f"{recommendation} {market_spread}",
                    edge_percentage=edge,
                    confidence="HIGH" if edge > 0.06 else "MEDIUM",
                    reasoning=f"Model: {our_spread:.1f}, Market: {market_spread}",
                    alert_level=alert_level,
                    expires_at=odds.game_time - timedelta(minutes=30),
                    sportsbooks=[book]
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_total_value(self, odds: LiveOddsData, prediction) -> List[ValueAlert]:
        """Analyze total betting value"""
        alerts = []
        
        our_total = prediction.final_total_points
        
        for book, market_total in odds.over_under.items():
            edge = abs(our_total - market_total) / market_total
            
            if edge > self.VALUE_THRESHOLD:
                recommendation = "OVER" if our_total > market_total else "UNDER"
                alert_level = "URGENT" if edge > self.URGENT_THRESHOLD else "HIGH"
                
                alert = ValueAlert(
                    game_id=odds.game_id,
                    bet_type="TOTAL",
                    recommendation=f"{recommendation} {market_total}",
                    edge_percentage=edge,
                    confidence="HIGH" if edge > 0.06 else "MEDIUM",
                    reasoning=f"Model: {our_total:.1f}, Market: {market_total}",
                    alert_level=alert_level,
                    expires_at=odds.game_time - timedelta(minutes=30),
                    sportsbooks=[book]
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_moneyline_value(self, odds: LiveOddsData, prediction) -> List[ValueAlert]:
        """Analyze moneyline betting value"""
        alerts = []
        
        our_prob = prediction.final_win_probability
        
        for book in odds.home_ml.keys():
            home_ml = odds.home_ml[book]
            away_ml = odds.away_ml[book]
            
            # Convert odds to implied probability
            home_implied = self._ml_to_prob(home_ml)
            away_implied = self._ml_to_prob(away_ml)
            
            # Check for value on home team
            home_edge = our_prob - home_implied
            if home_edge > self.VALUE_THRESHOLD:
                alert = ValueAlert(
                    game_id=odds.game_id,
                    bet_type="MONEYLINE",
                    recommendation=f"HOME {home_ml}",
                    edge_percentage=home_edge,
                    confidence="HIGH" if home_edge > 0.08 else "MEDIUM",
                    reasoning=f"Model: {our_prob:.1%}, Implied: {home_implied:.1%}",
                    alert_level="URGENT" if home_edge > 0.1 else "HIGH",
                    expires_at=odds.game_time - timedelta(minutes=30),
                    sportsbooks=[book]
                )
                alerts.append(alert)
            
            # Check for value on away team
            away_edge = (1 - our_prob) - away_implied
            if away_edge > self.VALUE_THRESHOLD:
                alert = ValueAlert(
                    game_id=odds.game_id,
                    bet_type="MONEYLINE",
                    recommendation=f"AWAY {away_ml}",
                    edge_percentage=away_edge,
                    confidence="HIGH" if away_edge > 0.08 else "MEDIUM",
                    reasoning=f"Model: {1-our_prob:.1%}, Implied: {away_implied:.1%}",
                    alert_level="URGENT" if away_edge > 0.1 else "HIGH",
                    expires_at=odds.game_time - timedelta(minutes=30),
                    sportsbooks=[book]
                )
                alerts.append(alert)
        
        return alerts
    
    def _ml_to_prob(self, ml_odds: float) -> float:
        """Convert moneyline odds to implied probability"""
        if ml_odds > 0:
            return 100 / (ml_odds + 100)
        else:
            return abs(ml_odds) / (abs(ml_odds) + 100)
    
    def send_alerts(self, alerts: List[ValueAlert]):
        """Send alerts for value opportunities"""
        for alert in alerts:
            alert_key = f"{alert.game_id}_{alert.bet_type}_{alert.recommendation}"
            
            # Avoid duplicate alerts
            if alert_key not in self.sent_alerts:
                self._send_alert(alert)
                self.sent_alerts.add(alert_key)
                
                # Store in database
                self._store_alert(alert)
    
    def _send_alert(self, alert: ValueAlert):
        """Send individual alert"""
        message = f"""
🏀 {alert.alert_level} VALUE ALERT 🏀

Game: {alert.game_id}
Bet: {alert.bet_type} - {alert.recommendation}
Edge: {alert.edge_percentage:.1%}
Confidence: {alert.confidence}
Reasoning: {alert.reasoning}
Expires: {alert.expires_at.strftime('%H:%M')}
Sportsbooks: {', '.join(alert.sportsbooks)}
        """
        
        print(message)  # Replace with actual notification system
        self.logger.info(f"Alert sent: {alert.bet_type} {alert.recommendation} ({alert.edge_percentage:.1%} edge)")
    
    def _store_alert(self, alert: ValueAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO value_alerts 
            (timestamp, game_id, bet_type, recommendation, edge_percentage, 
             confidence, alert_level, reasoning, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(), alert.game_id, alert.bet_type, alert.recommendation,
            alert.edge_percentage, alert.confidence, alert.alert_level,
            alert.reasoning, alert.expires_at
        ))
        
        conn.commit()
        conn.close()
    
    async def monitor_continuously(self, duration_hours: int = 8):
        """Run continuous monitoring"""
        self.logger.info(f"Starting continuous monitoring for {duration_hours} hours")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Fetch live odds
                odds_data = await self.fetch_live_odds()
                
                if odds_data:
                    # Analyze for opportunities
                    alerts = self.analyze_betting_opportunities(odds_data)
                    
                    # Send alerts
                    if alerts:
                        self.send_alerts(alerts)
                        self.logger.info(f"Found {len(alerts)} value opportunities")
                    
                    # Store odds data
                    self._store_odds_data(odds_data)
                
                # Wait before next check
                await asyncio.sleep(self.MONITORING_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _store_odds_data(self, odds_data: Dict[str, LiveOddsData]):
        """Store odds data for historical analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game_id, odds in odds_data.items():
            for book in odds.home_spread.keys():
                cursor.execute('''
                    INSERT INTO live_odds_history 
                    (timestamp, game_id, home_team, away_team, sportsbook,
                     spread_home, spread_away, ml_home, ml_away, total,
                     over_odds, under_odds, betting_volume, sharp_indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    odds.timestamp, game_id, odds.home_team, odds.away_team, book,
                    odds.home_spread.get(book), odds.away_spread.get(book),
                    odds.home_ml.get(book), odds.away_ml.get(book),
                    odds.over_under.get(book), odds.over_odds.get(book),
                    odds.under_odds.get(book), odds.betting_volume.get(book),
                    json.dumps(odds.sharp_money_indicators.get(book, {}))
                ))
        
        conn.commit()
        conn.close()

# Testing and demonstration
async def main():
    """Test the enhanced monitoring system"""
    monitor = EnhancedLiveMonitor()
    
    print("🏀 Enhanced Live Monitor Demo")
    print("=" * 50)
    
    # Simulate one monitoring cycle
    odds_data = await monitor.fetch_live_odds()
    print(f"Fetched odds for {len(odds_data)} games")
    
    alerts = monitor.analyze_betting_opportunities(odds_data)
    print(f"Found {len(alerts)} value opportunities")
    
    if alerts:
        monitor.send_alerts(alerts)
        print("\nValue alerts sent!")
    
    print("\nMonitoring system ready!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    asyncio.run(main())