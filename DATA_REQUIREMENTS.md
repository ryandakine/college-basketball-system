# ⚠️ CRITICAL: REAL DATA REQUIREMENTS ⚠️

## NO SYNTHETIC, MOCK, OR FAKE DATA

**This system requires 100% REAL DATA for accurate predictions. Using synthetic or mock data WILL destroy prediction accuracy and cause financial losses.**

## Required Data Sources

### 1. Team Statistics (REQUIRED)

For each team, you need REAL historical data:

```python
team_stats = {
    'offensive_efficiency': float,  # Points per 100 possessions (from KenPom/Barttorvik)
    'defensive_efficiency': float,  # Points allowed per 100 possessions
    'tempo': float,                 # Possessions per game
    'wins': int,                    # Season wins
    'losses': int,                  # Season losses
    'kenpom_rating': float,         # KenPom efficiency margin
    'net_rating': float,            # Point differential per game
    'avg_points_scored': float,
    'avg_points_allowed': float,
    'recent_form': float            # Win% in last 5 games
}
```

**Data Sources:**
- KenPom.com (subscription required, most accurate)
- Barttorvik.com (free alternative)
- Sports Reference CBB
- ESPN Stats

### 2. Injury Data (REQUIRED)

**NO mock injury data.** Use real injury reports:

```python
injury_report = {
    'player_name': str,
    'position': str,  # 'PG', 'SG', 'SF', 'PF', 'C'
    'injury_type': str,
    'status': str,    # 'Out', 'Doubtful', 'Questionable', 'Probable'
    'minutes_per_game': float,
    'points_per_game': float,
    'player_efficiency_rating': float
}
```

**Data Sources:**
- Team official injury reports
- ESPN injury reports  
- CBS Sports injury updates
- RotoWire college basketball injuries

### 3. Historical Matchup Data (REQUIRED)

**Real game results only:**

```python
game_result = {
    'date': str,
    'team_a': str,
    'team_b': str,
    'team_a_score': int,
    'team_b_score': int,
    'venue': str,
    'neutral_site': bool,
    'tournament_game': bool,
    'tempo': float
}
```

**Data Sources:**
- Sports Reference game logs
- KenPom game data
- ESPN box scores

### 4. Referee Data (OPTIONAL but recommended)

**Real referee statistics:**

```python
referee_profile = {
    'referee_name': str,
    'games_worked': int,
    'avg_fouls_per_game': float,
    'home_foul_bias': float,        # Calculate from real game data
    'pace_factor': float,           # Calculate from real game data
    'experience_years': int,
    'conference_specialty': str
}
```

**Data Sources:**
- collegehoopswatch.com
- Manual tracking from box scores
- KenPom referee stats

### 5. Venue Data (OPTIONAL but recommended)

**Real venue statistics:**

```python
venue_profile = {
    'venue_name': str,
    'home_team': str,
    'capacity': int,
    'home_win_percentage': float,  # Calculate from real games
    'avg_total_points': float,     # Calculate from real games
    'upset_frequency': float       # Calculate from real games
}
```

**Data Sources:**
- Team websites
- Sports Reference venue data
- Manual tracking

## Data Import Process

### Step 1: Import Team Statistics

```python
from core_basketball_prediction_engine import CoreBasketballPredictionEngine

engine = CoreBasketballPredictionEngine()

# Example with REAL data from KenPom
game_data = {
    'game_id': 'duke_unc_20251115',
    'date': '2025-11-15',
    'home_team': 'Duke',
    'away_team': 'North Carolina',
    'home_team_stats': {
        'offensive_efficiency': 118.5,  # From KenPom
        'defensive_efficiency': 95.2,   # From KenPom
        'tempo': 72.3                    # From KenPom
    },
    'away_team_stats': {
        'offensive_efficiency': 115.8,
        'defensive_efficiency': 98.1,
        'tempo': 69.8
    },
    # ... betting lines, odds, etc
}

prediction = engine.generate_comprehensive_prediction(game_data)
```

### Step 2: Import Historical Data

```python
from basketball_rag_system import BasketballRAGSystem

rag = BasketballRAGSystem()

# Add REAL game results as they happen
rag.add_game_result(
    team_a="Duke",
    team_b="North Carolina",
    team_a_score=78,
    team_b_score=75,
    game_date="2024-02-03",
    venue="Cameron Indoor Stadium",
    neutral_site=False,
    tournament_game=False,
    tempo=68.5
)
```

### Step 3: Import Referee Data (Optional)

```python
# Import REAL referee data
referee_data = [
    {
        'referee_name': 'John Higgins',
        'games_worked': 250,
        'avg_fouls_per_game': 42.5,
        'avg_points_per_game': 145.2,
        'home_foul_bias': -1.2,
        'pace_factor': 0.98,
        'technical_foul_rate': 0.08,
        'experience_years': 25,
        'conference_specialty': 'ACC,SEC',
        'tournament_games': 45
    }
]

rag.import_referee_data(referee_data)
```

### Step 4: Import Venue Data (Optional)

```python
venue_data = [
    {
        'venue_name': 'Cameron Indoor Stadium',
        'home_team': 'Duke',
        'capacity': 9314,
        'home_win_percentage': 0.842,
        'avg_attendance': 9100,
        'avg_total_points': 152.3,
        'upset_frequency': 0.15,
        'characteristics': 'Loud,Intimate,Historic'
    }
]

rag.import_venue_data(venue_data)
```

## Data Validation

Before using any data, validate it:

```python
def validate_team_stats(stats):
    """Ensure team stats are realistic"""
    assert 80 < stats['offensive_efficiency'] < 140, "Invalid offensive efficiency"
    assert 80 < stats['defensive_efficiency'] < 140, "Invalid defensive efficiency"
    assert 55 < stats['tempo'] < 85, "Invalid tempo"
    assert 0 <= stats['wins'] <= 40, "Invalid wins"
    assert 0 <= stats['losses'] <= 40, "Invalid losses"
```

## Critical Warnings

### ❌ DO NOT USE:
- Synthetic data
- Mock data
- Sample data
- Randomly generated data
- Placeholder values
- Default values (except as fallbacks when real data unavailable)

### ✅ ONLY USE:
- Real game results
- Real team statistics from KenPom/Barttorvik
- Real injury reports from official sources
- Real betting lines from sportsbooks
- Real referee data (if tracking)

## Data Freshness

**All data MUST be current:**

- **Team stats**: Updated daily during season
- **Injury reports**: Updated within 24 hours
- **Betting lines**: Real-time or as close as possible
- **Historical results**: Added immediately after games
- **Referee data**: Updated after each game worked

## Consequences of Bad Data

Using synthetic or outdated data will cause:

1. **Inaccurate predictions** - Wrong win probabilities
2. **Bad betting recommendations** - False edges
3. **Financial losses** - Betting on bad information
4. **Overconfident predictions** - Calibration breaks
5. **System unreliability** - Trust in system destroyed

## Data Collection Scripts

You need to create scripts to scrape/import from:

1. **KenPom API** (if available) or web scraping
2. **ESPN API** for scores and injury reports
3. **Sportsbook APIs** for real-time lines
4. **Sports Reference** for historical data

## Minimum Data Requirements

**To start making predictions, you MUST have:**

- [x] Team efficiency ratings (KenPom or Barttorvik)
- [x] Current season records
- [x] Current injury reports
- [x] Real betting lines

**Recommended additions:**

- [ ] Historical head-to-head results (50+ games minimum)
- [ ] Referee data (20+ games per referee)
- [ ] Venue data (full season minimum)
- [ ] Player-level statistics

## Testing with Real Data

Before going live:

1. Backtest with last season's data
2. Validate prediction accuracy on known results
3. Check calibration curves
4. Verify betting recommendations would be profitable

## Questions?

If you don't have access to real data sources:

1. **Subscribe to KenPom** ($19.95/year) - ESSENTIAL
2. **Use Barttorvik** (free alternative)
3. **Scrape ESPN** for injury reports
4. **Track games manually** if needed

**DO NOT** proceed with predictions until you have real data sources configured.
