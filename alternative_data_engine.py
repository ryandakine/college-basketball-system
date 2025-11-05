#!/usr/bin/env python3
"""
Alternative Data Integration Engine for College Basketball
=========================================================

Unique data sources for betting edges:
- Social media sentiment analysis
- Weather impact for outdoor/dome venues
- Team travel patterns and jet lag
- Academic calendar impacts (finals week, breaks)
- Recruiting news and team chemistry
- Injury rumor detection
- Coaching staff changes
- Fan attendance patterns
- Conference tournament implications
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
import requests
import asyncio
import aiohttp
import re
from textblob import TextBlob
import tweepy
from bs4 import BeautifulSoup

# Optional imports for advanced features
try:
    import yfinance as yf
    import alpha_vantage
    FINANCIAL_DATA_AVAILABLE = True
except ImportError:
    FINANCIAL_DATA_AVAILABLE = False

@dataclass
class SentimentData:
    """Social media sentiment data"""
    team: str
    timestamp: datetime
    source: str  # 'twitter', 'reddit', 'forums'
    
    # Sentiment metrics
    overall_sentiment: float  # -1 to 1
    confidence_sentiment: float  # -1 to 1
    volume_mentions: int
    engagement_score: float
    
    # Key topics
    trending_topics: List[str]
    injury_rumors: List[str]
    coaching_concerns: List[str]
    
    # Comparative analysis
    vs_opponent_sentiment: float
    historical_baseline: float

@dataclass
class TravelImpact:
    """Team travel and fatigue analysis"""
    team: str
    game_date: datetime
    
    # Travel metrics
    distance_traveled: float  # miles
    time_zones_crossed: int
    travel_method: str  # 'flight', 'bus', 'charter'
    arrival_time: datetime
    
    # Fatigue indicators
    days_since_last_game: int
    games_in_last_week: int
    consecutive_road_games: int
    
    # Recovery factors
    practice_time_available: float  # hours
    sleep_disruption_score: float  # 0-1
    jet_lag_severity: float  # 0-1
    
    # Historical performance
    travel_performance_rating: float  # vs baseline

@dataclass
class AcademicImpact:
    """Academic calendar impact on performance"""
    team: str
    date: datetime
    
    # Academic factors
    is_finals_week: bool
    is_exam_period: bool
    is_break_period: bool
    semester_week: int  # Week of semester
    
    # Historical patterns
    academic_performance_drop: float  # Expected performance change
    attendance_impact: float
    player_availability_risk: float
    
    # School-specific factors
    academic_rigor: float  # School's academic difficulty rating
    athlete_support_quality: float

@dataclass
class WeatherImpact:
    """Weather impact analysis"""
    venue: str
    game_date: datetime
    
    # Weather conditions
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    
    # Venue factors
    is_outdoor: bool
    is_dome: bool
    hvac_quality: float
    
    # Impact analysis
    fan_attendance_impact: float
    travel_disruption_risk: float
    player_comfort_impact: float

@dataclass
class RecruitingNews:
    """Recruiting and team chemistry news"""
    team: str
    timestamp: datetime
    
    # News type
    news_type: str  # 'transfer', 'recruiting', 'dismissal', 'injury'
    player_involved: str
    impact_severity: float  # 0-1
    
    # Team chemistry impact
    chemistry_impact: float  # -1 to 1
    locker_room_stability: float  # 0-1
    coaching_stability: float  # 0-1
    
    # Performance prediction
    short_term_impact: float  # Next 5 games
    long_term_impact: float   # Rest of season

class AlternativeDataEngine:
    """Alternative data integration for unique betting edges"""
    
    def __init__(self, db_path: str = "alternative_data.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # API configurations (would need real API keys)
        self.apis = {
            'twitter': {'bearer_token': 'YOUR_TWITTER_BEARER_TOKEN'},
            'weather': {'api_key': 'YOUR_WEATHER_API_KEY'},
            'news': {'api_key': 'YOUR_NEWS_API_KEY'}
        }
        
        # Data collection parameters
        self.SENTIMENT_KEYWORDS = {
            'positive': ['confident', 'strong', 'ready', 'motivated', 'healthy', 'focused'],
            'negative': ['injured', 'tired', 'distracted', 'struggling', 'suspended', 'transfer'],
            'confidence': ['we got this', 'easy win', 'ready to dominate', 'bring it on'],
            'concern': ['worried about', 'not sure', 'hope we can', 'might struggle']
        }
        
        # Impact weights
        self.IMPACT_WEIGHTS = {
            'sentiment': 0.15,
            'travel': 0.20,
            'academic': 0.10,
            'weather': 0.05,
            'recruiting': 0.25,
            'injuries': 0.25
        }
    
    def _init_database(self):
        """Initialize alternative data database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sentiment data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                timestamp DATETIME,
                source TEXT,
                overall_sentiment REAL,
                confidence_sentiment REAL,
                volume_mentions INTEGER,
                engagement_score REAL,
                vs_opponent_sentiment REAL,
                historical_baseline REAL
            )
        ''')
        
        # Travel impact
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS travel_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                game_date DATETIME,
                distance_traveled REAL,
                time_zones_crossed INTEGER,
                travel_method TEXT,
                days_since_last_game INTEGER,
                games_in_last_week INTEGER,
                jet_lag_severity REAL,
                travel_performance_rating REAL
            )
        ''')
        
        # Academic impact
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS academic_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                date DATETIME,
                is_finals_week BOOLEAN,
                is_exam_period BOOLEAN,
                semester_week INTEGER,
                academic_performance_drop REAL,
                attendance_impact REAL,
                academic_rigor REAL
            )
        ''')
        
        # Weather data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT,
                game_date DATETIME,
                temperature REAL,
                humidity REAL,
                precipitation REAL,
                is_outdoor BOOLEAN,
                fan_attendance_impact REAL,
                travel_disruption_risk REAL
            )
        ''')
        
        # Recruiting news
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recruiting_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                timestamp DATETIME,
                news_type TEXT,
                player_involved TEXT,
                impact_severity REAL,
                chemistry_impact REAL,
                short_term_impact REAL,
                long_term_impact REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def collect_sentiment_data(self, teams: List[str], days_back: int = 7) -> List[SentimentData]:
        """Collect social media sentiment for teams"""
        sentiment_data = []
        
        for team in teams:
            # Twitter sentiment
            twitter_sentiment = await self._get_twitter_sentiment(team, days_back)
            if twitter_sentiment:
                sentiment_data.append(twitter_sentiment)
            
            # Reddit sentiment
            reddit_sentiment = await self._get_reddit_sentiment(team, days_back)
            if reddit_sentiment:
                sentiment_data.append(reddit_sentiment)
            
            # Sports forum sentiment
            forum_sentiment = await self._get_forum_sentiment(team, days_back)
            if forum_sentiment:
                sentiment_data.append(forum_sentiment)
        
        # Store in database
        for data in sentiment_data:
            self._store_sentiment_data(data)
        
        return sentiment_data
    
    async def _get_twitter_sentiment(self, team: str, days_back: int) -> Optional[SentimentData]:
        """Get Twitter sentiment for a team"""
        try:
            # This would use Twitter API v2 - simulating for demo
            # In practice, you'd collect tweets about the team
            
            # Simulate tweet collection and sentiment analysis
            tweets = self._simulate_tweets(team)
            
            # Analyze sentiment
            overall_sentiment = 0.0
            confidence_sentiment = 0.0
            volume = len(tweets)
            engagement = 0.0
            
            injury_rumors = []
            trending_topics = []
            
            for tweet in tweets:
                # Sentiment analysis using TextBlob
                blob = TextBlob(tweet['text'])
                overall_sentiment += blob.sentiment.polarity
                engagement += tweet.get('engagement', 0)
                
                # Check for injury rumors
                if any(word in tweet['text'].lower() for word in ['injured', 'hurt', 'out', 'questionable']):
                    injury_rumors.append(tweet['text'][:100])
                
                # Check for confidence indicators
                if any(word in tweet['text'].lower() for word in self.SENTIMENT_KEYWORDS['confidence']):
                    confidence_sentiment += 0.3
                elif any(word in tweet['text'].lower() for word in self.SENTIMENT_KEYWORDS['concern']):
                    confidence_sentiment -= 0.3
            
            if volume > 0:
                overall_sentiment /= volume
                confidence_sentiment = max(-1, min(1, confidence_sentiment / volume))
                engagement /= volume
            
            return SentimentData(
                team=team,
                timestamp=datetime.now(),
                source='twitter',
                overall_sentiment=overall_sentiment,
                confidence_sentiment=confidence_sentiment,
                volume_mentions=volume,
                engagement_score=engagement,
                trending_topics=trending_topics[:5],
                injury_rumors=injury_rumors[:3],
                coaching_concerns=[],
                vs_opponent_sentiment=0.0,  # Would compare with opponent
                historical_baseline=0.0     # Would use historical data
            )
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment for {team}: {e}")
            return None
    
    def _simulate_tweets(self, team: str) -> List[Dict[str, Any]]:
        """Simulate tweet data for demo"""
        tweets = [
            {
                'text': f'{team} looking strong in practice this week. Ready for the big game!',
                'engagement': 150,
                'sentiment': 0.8
            },
            {
                'text': f'Heard {team} star player might be questionable for tonight. Hope he plays.',
                'engagement': 85,
                'sentiment': -0.2
            },
            {
                'text': f'{team} fans are confident about this matchup. Home court advantage!',
                'engagement': 200,
                'sentiment': 0.6
            },
            {
                'text': f'Not sure about {team} defense lately. Giving up too many easy buckets.',
                'engagement': 95,
                'sentiment': -0.4
            }
        ]
        return tweets
    
    async def _get_reddit_sentiment(self, team: str, days_back: int) -> Optional[SentimentData]:
        """Get Reddit sentiment for a team"""
        # Similar to Twitter but would scrape Reddit
        # For demo, return simplified sentiment
        return SentimentData(
            team=team,
            timestamp=datetime.now(),
            source='reddit',
            overall_sentiment=np.random.normal(0.1, 0.3),  # Slightly positive bias
            confidence_sentiment=np.random.normal(0.0, 0.4),
            volume_mentions=np.random.randint(20, 100),
            engagement_score=np.random.uniform(0.3, 0.8),
            trending_topics=['defense', 'shooting', 'coach'],
            injury_rumors=[],
            coaching_concerns=[],
            vs_opponent_sentiment=0.0,
            historical_baseline=0.0
        )
    
    async def _get_forum_sentiment(self, team: str, days_back: int) -> Optional[SentimentData]:
        """Get sports forum sentiment"""
        # Would scrape sites like 247Sports, Rivals, etc.
        return SentimentData(
            team=team,
            timestamp=datetime.now(),
            source='forums',
            overall_sentiment=np.random.normal(0.05, 0.25),
            confidence_sentiment=np.random.normal(0.0, 0.35),
            volume_mentions=np.random.randint(10, 50),
            engagement_score=np.random.uniform(0.4, 0.9),
            trending_topics=['recruiting', 'lineup', 'injuries'],
            injury_rumors=[],
            coaching_concerns=[],
            vs_opponent_sentiment=0.0,
            historical_baseline=0.0
        )
    
    def analyze_travel_impact(self, team: str, game_date: datetime, 
                            opponent_venue: str, previous_game_venue: str) -> TravelImpact:
        """Analyze travel impact on team performance"""
        
        # Calculate travel distance (simplified - would use real geo data)
        distance = self._calculate_travel_distance(previous_game_venue, opponent_venue)
        
        # Time zone analysis
        time_zones = self._calculate_timezone_difference(previous_game_venue, opponent_venue)
        
        # Determine travel method
        travel_method = 'flight' if distance > 500 else 'bus'
        
        # Calculate jet lag severity
        jet_lag = min(1.0, time_zones / 3.0) if time_zones > 0 else 0.0
        
        # Simulate historical travel performance
        travel_rating = np.random.normal(0.85, 0.15)  # Teams typically perform 85% on road
        
        # Academic calendar check
        is_finals = self._is_finals_week(team, game_date)
        
        return TravelImpact(
            team=team,
            game_date=game_date,
            distance_traveled=distance,
            time_zones_crossed=time_zones,
            travel_method=travel_method,
            arrival_time=game_date - timedelta(hours=max(1, distance/500)),
            days_since_last_game=3,  # Would calculate from schedule
            games_in_last_week=2,    # Would calculate from schedule
            consecutive_road_games=1,
            practice_time_available=max(0, 48 - distance/100),
            sleep_disruption_score=jet_lag * 0.8,
            jet_lag_severity=jet_lag,
            travel_performance_rating=travel_rating * (1 - jet_lag * 0.1)
        )
    
    def _calculate_travel_distance(self, venue1: str, venue2: str) -> float:
        """Calculate travel distance between venues"""
        # Simplified - would use real geocoding
        venue_distances = {
            ('Duke', 'UNC'): 12,
            ('Duke', 'Kentucky'): 350,
            ('Kansas', 'Texas'): 450,
            ('UCLA', 'Arizona'): 500,
            ('Duke', 'Stanford'): 2400,
        }
        
        key = (venue1, venue2)
        if key in venue_distances:
            return venue_distances[key]
        elif (venue2, venue1) in venue_distances:
            return venue_distances[(venue2, venue1)]
        else:
            return np.random.uniform(100, 1000)  # Default random distance
    
    def _calculate_timezone_difference(self, venue1: str, venue2: str) -> int:
        """Calculate timezone difference"""
        # Simplified timezone mapping
        timezone_map = {
            'Duke': -5, 'UNC': -5, 'Kentucky': -5,  # Eastern
            'Kansas': -6, 'Texas': -6,              # Central
            'Arizona': -7, 'UCLA': -8, 'Stanford': -8  # Mountain/Pacific
        }
        
        tz1 = timezone_map.get(venue1, -5)
        tz2 = timezone_map.get(venue2, -5)
        
        return abs(tz1 - tz2)
    
    def _is_finals_week(self, team: str, date: datetime) -> bool:
        """Check if date falls during finals week"""
        # Typical finals weeks: mid-December and early May
        if date.month == 12 and 10 <= date.day <= 20:
            return True
        elif date.month == 5 and 1 <= date.day <= 15:
            return True
        return False
    
    def analyze_academic_impact(self, team: str, game_date: datetime) -> AcademicImpact:
        """Analyze academic calendar impact"""
        
        # Determine academic period
        is_finals = self._is_finals_week(team, game_date)
        is_exams = self._is_exam_period(team, game_date)
        is_break = self._is_break_period(team, game_date)
        
        # Calculate semester week
        semester_start = self._get_semester_start(game_date)
        semester_week = (game_date - semester_start).days // 7
        
        # School-specific factors
        academic_rigor = self._get_academic_rigor(team)
        
        # Historical performance impact
        performance_drop = 0.0
        if is_finals:
            performance_drop = 0.15 * academic_rigor  # 15% drop for rigorous schools
        elif is_exams:
            performance_drop = 0.08 * academic_rigor
        
        return AcademicImpact(
            team=team,
            date=game_date,
            is_finals_week=is_finals,
            is_exam_period=is_exams,
            is_break_period=is_break,
            semester_week=semester_week,
            academic_performance_drop=performance_drop,
            attendance_impact=performance_drop * 0.5,  # Attendance drops less than performance
            player_availability_risk=0.1 if is_finals else 0.0,
            academic_rigor=academic_rigor,
            athlete_support_quality=0.8  # Most schools have good athlete support
        )
    
    def _is_exam_period(self, team: str, date: datetime) -> bool:
        """Check if in exam period"""
        # Broader exam periods
        if date.month == 12 and 5 <= date.day <= 22:
            return True
        elif date.month == 5 and 1 <= date.day <= 20:
            return True
        elif date.month in [10, 3] and 15 <= date.day <= 25:  # Mid-terms
            return True
        return False
    
    def _is_break_period(self, team: str, date: datetime) -> bool:
        """Check if during break period"""
        # Winter break, spring break
        if (date.month == 12 and date.day >= 23) or date.month == 1:
            return True
        elif date.month == 3 and 15 <= date.day <= 30:  # Spring break varies
            return True
        return False
    
    def _get_semester_start(self, date: datetime) -> datetime:
        """Get semester start date"""
        if date.month >= 8:  # Fall semester
            return datetime(date.year, 8, 20)
        else:  # Spring semester
            return datetime(date.year, 1, 15)
    
    def _get_academic_rigor(self, team: str) -> float:
        """Get academic rigor score for school"""
        # Simplified academic rigor mapping
        rigor_map = {
            'Duke': 0.95, 'Stanford': 0.95, 'Northwestern': 0.90,
            'UNC': 0.75, 'Michigan': 0.80, 'UCLA': 0.80,
            'Kentucky': 0.60, 'Kansas': 0.65, 'Arizona': 0.70
        }
        return rigor_map.get(team, 0.70)  # Default moderate rigor
    
    async def collect_weather_data(self, venue: str, game_date: datetime) -> WeatherImpact:
        """Collect weather data for venue"""
        
        # This would use a real weather API
        # For demo, simulate weather data
        
        # Check if venue is outdoor/dome
        outdoor_venues = ['Rose Bowl', 'LA Coliseum']  # Very few for basketball
        dome_venues = ['Carrier Dome', 'Alamodome']
        
        is_outdoor = venue in outdoor_venues
        is_dome = venue in dome_venues
        
        # Simulate weather
        temp = np.random.normal(45, 15)  # Winter temps
        humidity = np.random.uniform(30, 80)
        precip = np.random.exponential(0.1)  # Usually no precipitation
        
        # Calculate impacts
        fan_impact = 0.0
        travel_impact = 0.0
        
        if is_outdoor:
            if temp < 32 or precip > 0.5:  # Cold or wet
                fan_impact = -0.2  # 20% attendance drop
                travel_impact = 0.3   # 30% travel disruption risk
        
        if precip > 1.0:  # Heavy precipitation
            travel_impact = min(1.0, travel_impact + 0.4)
        
        return WeatherImpact(
            venue=venue,
            game_date=game_date,
            temperature=temp,
            humidity=humidity,
            precipitation=precip,
            wind_speed=np.random.uniform(0, 20),
            is_outdoor=is_outdoor,
            is_dome=is_dome,
            hvac_quality=0.8 if not is_outdoor else 0.0,
            fan_attendance_impact=fan_impact,
            travel_disruption_risk=travel_impact,
            player_comfort_impact=0.1 if is_outdoor and (temp < 40 or temp > 85) else 0.0
        )
    
    async def collect_recruiting_news(self, teams: List[str], days_back: int = 30) -> List[RecruitingNews]:
        """Collect recruiting and team chemistry news"""
        news_items = []
        
        for team in teams:
            # This would scrape recruiting websites, news sites
            # For demo, simulate some news
            
            # Simulate transfer portal activity
            if np.random.random() < 0.1:  # 10% chance of transfer news
                news_items.append(RecruitingNews(
                    team=team,
                    timestamp=datetime.now() - timedelta(days=np.random.randint(1, days_back)),
                    news_type='transfer',
                    player_involved='Key Player',
                    impact_severity=0.7,
                    chemistry_impact=-0.3,
                    locker_room_stability=0.6,
                    coaching_stability=0.8,
                    short_term_impact=-0.15,
                    long_term_impact=-0.05
                ))
            
            # Simulate injury news
            if np.random.random() < 0.15:  # 15% chance of injury news
                news_items.append(RecruitingNews(
                    team=team,
                    timestamp=datetime.now() - timedelta(days=np.random.randint(1, 7)),
                    news_type='injury',
                    player_involved='Starter',
                    impact_severity=0.5,
                    chemistry_impact=-0.1,
                    locker_room_stability=0.8,
                    coaching_stability=0.9,
                    short_term_impact=-0.25,
                    long_term_impact=-0.10
                ))
        
        # Store news items
        for news in news_items:
            self._store_recruiting_news(news)
        
        return news_items
    
    def calculate_alternative_data_edge(self, home_team: str, away_team: str, 
                                      game_date: datetime, venue: str) -> Dict[str, float]:
        """Calculate betting edge from alternative data"""
        
        edges = {}
        
        # Get recent sentiment data
        home_sentiment = self._get_recent_sentiment(home_team)
        away_sentiment = self._get_recent_sentiment(away_team)
        
        if home_sentiment and away_sentiment:
            sentiment_edge = (home_sentiment.overall_sentiment - away_sentiment.overall_sentiment) * self.IMPACT_WEIGHTS['sentiment']
            edges['sentiment'] = sentiment_edge
        
        # Travel impact
        home_travel = self.analyze_travel_impact(home_team, game_date, venue, "Previous Venue")
        away_travel = self.analyze_travel_impact(away_team, game_date, venue, "Previous Venue")
        
        travel_edge = (away_travel.jet_lag_severity - home_travel.jet_lag_severity) * self.IMPACT_WEIGHTS['travel']
        edges['travel'] = travel_edge
        
        # Academic impact
        home_academic = self.analyze_academic_impact(home_team, game_date)
        away_academic = self.analyze_academic_impact(away_team, game_date)
        
        academic_edge = (away_academic.academic_performance_drop - home_academic.academic_performance_drop) * self.IMPACT_WEIGHTS['academic']
        edges['academic'] = academic_edge
        
        # Recruiting news impact
        home_recruiting = self._get_recent_recruiting_impact(home_team)
        away_recruiting = self._get_recent_recruiting_impact(away_team)
        
        recruiting_edge = (away_recruiting - home_recruiting) * self.IMPACT_WEIGHTS['recruiting']
        edges['recruiting'] = recruiting_edge
        
        # Total alternative data edge
        total_edge = sum(edges.values())
        edges['total'] = total_edge
        
        return edges
    
    def _get_recent_sentiment(self, team: str) -> Optional[SentimentData]:
        """Get most recent sentiment data for team"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sentiment_data 
            WHERE team = ? AND timestamp >= ?
            ORDER BY timestamp DESC LIMIT 1
        '''
        
        cutoff = datetime.now() - timedelta(days=3)
        result = conn.execute(query, (team, cutoff)).fetchone()
        conn.close()
        
        if result:
            return SentimentData(
                team=result[1],
                timestamp=datetime.fromisoformat(result[2]),
                source=result[3],
                overall_sentiment=result[4],
                confidence_sentiment=result[5],
                volume_mentions=result[6],
                engagement_score=result[7],
                trending_topics=[],
                injury_rumors=[],
                coaching_concerns=[],
                vs_opponent_sentiment=result[8],
                historical_baseline=result[9]
            )
        return None
    
    def _get_recent_recruiting_impact(self, team: str) -> float:
        """Get recent recruiting news impact"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT AVG(short_term_impact) FROM recruiting_news 
            WHERE team = ? AND timestamp >= ?
        '''
        
        cutoff = datetime.now() - timedelta(days=30)
        result = conn.execute(query, (team, cutoff)).fetchone()
        conn.close()
        
        return result[0] if result[0] is not None else 0.0
    
    def _store_sentiment_data(self, data: SentimentData):
        """Store sentiment data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_data 
            (team, timestamp, source, overall_sentiment, confidence_sentiment,
             volume_mentions, engagement_score, vs_opponent_sentiment, historical_baseline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.team, data.timestamp, data.source, data.overall_sentiment,
            data.confidence_sentiment, data.volume_mentions, data.engagement_score,
            data.vs_opponent_sentiment, data.historical_baseline
        ))
        
        conn.commit()
        conn.close()
    
    def _store_recruiting_news(self, news: RecruitingNews):
        """Store recruiting news in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recruiting_news 
            (team, timestamp, news_type, player_involved, impact_severity,
             chemistry_impact, short_term_impact, long_term_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            news.team, news.timestamp, news.news_type, news.player_involved,
            news.impact_severity, news.chemistry_impact, news.short_term_impact,
            news.long_term_impact
        ))
        
        conn.commit()
        conn.close()
    
    async def run_daily_collection(self, teams: List[str]):
        """Run daily alternative data collection"""
        self.logger.info("Starting daily alternative data collection...")
        
        # Collect sentiment data
        await self.collect_sentiment_data(teams)
        
        # Collect recruiting news
        await self.collect_recruiting_news(teams)
        
        # Log summary
        self.logger.info("Daily alternative data collection completed")

# Testing and demonstration
async def demo_alternative_data():
    """Demo the alternative data engine"""
    engine = AlternativeDataEngine()
    
    print("📊 Alternative Data Engine Demo")
    print("=" * 50)
    
    teams = ['Duke', 'UNC', 'Kansas', 'Kentucky']
    
    print(f"Collecting alternative data for {len(teams)} teams...")
    
    # Collect sentiment data
    sentiment_data = await engine.collect_sentiment_data(teams, days_back=7)
    print(f"📱 Collected sentiment data for {len(sentiment_data)} team sources")
    
    # Analyze travel impact
    travel_impact = engine.analyze_travel_impact('Duke', datetime.now() + timedelta(days=2), 'UNC', 'Home')
    print(f"✈️  Travel impact: {travel_impact.jet_lag_severity:.2f} jet lag severity")
    
    # Analyze academic impact
    academic_impact = engine.analyze_academic_impact('Duke', datetime.now())
    print(f"📚 Academic impact: {academic_impact.academic_performance_drop:.2%} performance drop")
    
    # Calculate total edge
    edges = engine.calculate_alternative_data_edge('Duke', 'UNC', datetime.now() + timedelta(days=2), 'UNC Arena')
    
    print(f"\n🎯 Alternative Data Edges:")
    for source, edge in edges.items():
        if source != 'total':
            print(f"• {source.title()}: {edge:+.1%}")
    print(f"• Total Alternative Edge: {edges['total']:+.1%}")
    
    print(f"\n✅ Alternative Data Engine operational!")

def main():
    """Main function for testing"""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_alternative_data())

if __name__ == "__main__":
    main()