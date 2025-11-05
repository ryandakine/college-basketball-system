# College Basketball Betting System

## Overview

This comprehensive college basketball betting system provides advanced analytics and predictions for college basketball games, with specialized features for March Madness tournament play. The system integrates multiple data sources and analytical models to generate betting recommendations with confidence metrics.

## 🏀 Key Features

### Core Prediction Engine
- **Game Outcome Predictions**: Win probability, point spreads, and total points
- **Tempo Analysis**: Possession-based predictions and pace adjustments  
- **Efficiency Metrics**: Offensive/defensive efficiency ratings (points per 100 possessions)
- **Tournament Context**: March Madness specific adjustments and upset potential
- **Conference Strength**: Multi-tier conference analysis and cross-conference comparisons

### Advanced Analytics
- **Strength of Schedule**: Quadrant-based resume analysis for tournament selection
- **Player Fatigue**: Tournament workload and rotation depth analysis
- **Team Depth**: Bench strength and injury replacement capabilities
- **Historical Performance**: Tournament experience and clutch game performance
- **Venue Effects**: Home court advantage, neutral site adjustments, crowd factors

### Betting Intelligence
- **Market Analysis**: Line movement and value identification
- **Edge Calculations**: Expected value and confidence-based unit sizing
- **Prop Betting**: Player and team props with statistical backing
- **Tournament Specials**: March Madness bracket and championship futures
- **Risk Management**: Bankroll management and maximum exposure limits

## 📊 System Architecture

```
college_basketball_system/
├── core_basketball_prediction_engine.py    # Main prediction system
├── basketball_analytics.py                 # Advanced analytics modules
├── basketball_injury_impact_system.py      # Injury tracking (basketball-specific)
├── basketball_depth_analysis.py           # Team depth and rotation analysis
├── basketball_model_training_pipeline.py   # ML model training
├── models/                                 # Trained models and data
├── data/                                  # Historical data and game logs
├── analysis/                              # Analysis results and reports
└── README.md                              # This documentation
```

## 🎯 Prediction Capabilities

### Game-Level Predictions
- **Win Probability**: Based on efficiency ratings, tempo, and situational factors
- **Point Spread**: Adjusted for home court, tournament context, and recent form
- **Total Points**: Tempo-based with offensive/defensive efficiency adjustments
- **Game Tempo**: Expected possessions with team pace analysis

### Tournament-Specific Features
- **Seeding Impact**: Historical seed performance and upset probability
- **Experience Factor**: Upperclassmen percentage and tournament history
- **Fatigue Analysis**: Back-to-back games and travel considerations
- **Bracket Context**: Regional placement and potential matchup paths

### Advanced Metrics
- **Quadrant Analysis**: Quality wins for tournament resume
- **KenPom Integration**: Efficiency-based rankings and projections
- **NET Ranking**: Tournament selection committee metrics
- **Conference Strength**: Multi-conference tournament bid projections

## 💡 Basketball-Specific Innovations

### Tempo-Based Modeling
Unlike other sports, college basketball predictions heavily weight tempo (possessions per game):
- Fast teams (72+ possessions) vs Slow teams (<65 possessions)
- Pace adjustments for tournament play (typically slower)
- Efficiency calculations per 100 possessions for fair comparison

### Tournament Context Awareness
- **March Madness Mode**: Upset potential based on seed differentials
- **Conference Tournament**: Familiarity factors and motivation adjustments  
- **Regular Season**: Standard predictive modeling

### Efficiency-First Approach
- **Offensive Efficiency**: Points scored per 100 possessions
- **Defensive Efficiency**: Points allowed per 100 possessions
- **Net Efficiency**: Primary ranking metric (better than simple win percentage)

### College-Specific Factors
- **Home Court Advantage**: Stronger than professional sports (3.5+ points)
- **Conference Strength Tiers**: Power 6 vs mid-major adjustments
- **Player Experience**: Upperclassmen impact on tournament performance
- **Coaching Experience**: Tournament history and big-game management

## 📈 Betting Applications

### Regular Season Betting
- **Spread Betting**: Point spread predictions with 4-6% edges identified
- **Total Betting**: Over/under with tempo and pace analysis
- **Moneyline**: Value identification in conference play
- **First Half**: Tempo adjustments for opening pace

### Tournament Betting
- **March Madness Spreads**: Upset potential and seed-based adjustments
- **Tournament Totals**: Pace typically slows in elimination games
- **Futures**: Championship odds and Final Four predictions
- **Bracket Challenges**: Optimal bracket construction strategies

### Prop Betting
- **Team Props**: Team total points, rebounds, three-pointers made
- **Player Props**: Points, rebounds, assists for key players
- **Game Props**: Margin of victory, overtime probability
- **Tournament Props**: Bracket busters, Cinderella teams

## 🔧 Technical Implementation

### Data Models

#### Game Context
```python
@dataclass
class GameContext:
    game_id: str
    home_team: str
    away_team: str
    tournament_context: str  # "regular_season", "conference_tournament", "march_madness"
    home_kenpom_rating: float
    away_kenpom_rating: float
    tournament_round: Optional[str]
    venue_capacity: int
    neutral_site: bool
```

#### Prediction Components
```python
@dataclass
class BasketballPredictionComponents:
    win_probability: float
    point_differential: float
    total_points: float
    tempo_prediction: float
    efficiency_predictions: Dict
    tournament_adjustments: Dict
    venue_adjustments: Dict
```

### Analytics Modules

#### Tempo Analysis
- Season-long pace tracking
- Opponent-adjusted tempo
- Tournament vs regular season pace
- Home/away tempo splits

#### Efficiency Metrics
- Offensive efficiency (points per 100 possessions)
- Defensive efficiency (points allowed per 100 possessions)
- Net efficiency ranking
- Four factors analysis (eFG%, TO%, OR%, FTR)

#### Strength of Schedule
- Quadrant-based analysis (NCAA tournament metrics)
- Conference vs non-conference strength
- Road/neutral/home game weighting
- Quality wins and bad losses tracking

## 🎲 Usage Examples

### Basic Prediction
```python
from core_basketball_prediction_engine import CoreBasketballPredictionEngine

engine = CoreBasketballPredictionEngine()

game_data = {
    'game_id': 'duke_unc_20250315',
    'home_team': 'Duke',
    'away_team': 'North Carolina', 
    'tournament_context': 'march_madness',
    'home_kenpom_rating': 85.2,
    'away_kenpom_rating': 78.6,
    'home_seed': 2,
    'away_seed': 4
}

prediction = engine.generate_comprehensive_prediction(game_data)
```

### Analytics Deep Dive
```python
from basketball_analytics import BasketballAnalytics

analytics = BasketballAnalytics()

# Tempo analysis
tempo = analytics.analyze_tempo('Duke', game_log, season_stats)

# Efficiency metrics  
efficiency = analytics.calculate_efficiency_metrics('Duke', game_log, season_stats)

# Tournament profile
tournament = analytics.analyze_tournament_profile('Duke', season_stats, historical_data)
```

## 📊 Performance Metrics

### Prediction Accuracy
- **ATS (Against The Spread)**: Target >52.4% for profitability
- **Totals**: Over/under accuracy with juice consideration
- **Moneyline**: Win percentage vs implied probability
- **Tournament**: Bracket performance and upset identification

### Betting Performance
- **ROI**: Return on investment by bet type and confidence level
- **Units**: Profit/loss tracking with Kelly Criterion sizing
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Bankroll management validation

### Model Validation
- **Calibration**: Predicted probabilities vs actual outcomes  
- **Brier Score**: Probabilistic accuracy measurement
- **Log Loss**: Penalized accuracy for confidence
- **Cross-Validation**: Out-of-sample testing protocols

## 🏆 March Madness Specialization

### Tournament Bracket Analysis
- **Seed Performance**: Historical analysis by seed matchup
- **Upset Probability**: Statistical likelihood by seed differential
- **Regional Strength**: Bracket balance and path difficulty
- **Sleeper Identification**: Mid-major teams with upset potential

### Selection Sunday Preparation
- **Bubble Analysis**: Last 4 In vs First 4 Out projections
- **Seeding Predictions**: 1-16 seed line projections
- **Bracket Matrix**: Regional assignments and potential matchups
- **Automatic Bids**: Conference tournament winner projections

### In-Tournament Adjustments
- **Fatigue Modeling**: Back-to-back game impacts
- **Motivation Factors**: Cinderella momentum vs pressure
- **Coaching Adjustments**: Timeout usage and strategic changes
- **Injury Updates**: Real-time roster change impacts

## 🔄 System Integration

### Data Sources
- **KenPom**: Efficiency ratings and tempo data
- **Sports Reference**: Historical statistics and game logs  
- **ESPN**: Real-time scores and injury reports
- **Injury Reports**: Team-specific news and roster updates
- **Vegas Lines**: Betting market data and line movement

### Real-Time Updates
- **Live Odds**: Continuous line monitoring for value
- **Injury News**: Player availability and impact analysis
- **Weather**: Outdoor venue considerations (rare in basketball)
- **Lineup Changes**: Starting five adjustments

### Performance Tracking
- **Prediction Logs**: Historical accuracy by model component
- **Betting Results**: P&L tracking by bet type and confidence
- **Model Drift**: Performance degradation detection
- **Recalibration**: Seasonal adjustments and improvements

## 🚀 Getting Started

### Installation
```bash
# Clone the basketball system
cd /home/ryan/college_basketball_system

# Install dependencies (Python 3.8+)
pip install numpy pandas scikit-learn sqlite3 logging

# Initialize databases
python core_basketball_prediction_engine.py
python basketball_analytics.py
```

### Quick Start
```python
# Generate your first prediction
from core_basketball_prediction_engine import CoreBasketballPredictionEngine

engine = CoreBasketballPredictionEngine()

# Sample game data (Duke vs UNC)
sample_game = {
    'game_id': 'sample_001',
    'home_team': 'Duke',
    'away_team': 'North Carolina',
    'home_kenpom_rating': 85.0,
    'away_kenpom_rating': 82.0,
    'tournament_context': 'regular_season'
}

prediction = engine.generate_comprehensive_prediction(sample_game)
print(f"Win Probability: {prediction.final_win_probability:.1%}")
print(f"Predicted Spread: {prediction.final_point_differential:+.1f}")
```

## 🎯 Profit Opportunities

### High-Value Markets
1. **Conference Tournaments**: Familiar teams, public bias opportunities
2. **March Madness First Weekend**: Maximum chaos, inefficient lines
3. **Mid-Major Conferences**: Less market attention, information edges
4. **Player Props**: Individual performance analytics advantages
5. **Live Betting**: Real-time adjustments as games develop

### Edge Identification
- **Line Shopping**: Multiple sportsbook comparison
- **Model Disagreements**: Significant deviations from market consensus
- **Public Bias**: Fade popular teams in tournament
- **Recency Bias**: Market overreaction to recent performance
- **Tournament Experience**: Undervalued veteran teams

## 📞 Support & Enhancement

### System Monitoring
- **Daily Performance**: Automated tracking and reporting
- **Model Health**: Statistical validation and drift detection  
- **Data Quality**: Source validation and completeness checks
- **Alert System**: Significant edge opportunities and system issues

### Future Enhancements
- **Player-Level Modeling**: Individual impact and matchup analysis
- **Advanced Shot Analytics**: Location-based efficiency modeling
- **Coaching Tendencies**: Strategic analysis and in-game adjustments
- **Transfer Portal Impact**: Mid-season roster change modeling
- **International Players**: European style adaptation analysis

---

## 🏀 Ready to Dominate March Madness?

This college basketball system combines the analytical rigor of professional sports betting with the unique characteristics of college basketball. From regular season conference play to the madness of March, every prediction is backed by comprehensive data analysis and proven modeling techniques.

**Key Advantages:**
- ✅ Tournament-specific modeling
- ✅ Tempo-based predictions  
- ✅ Conference strength adjustments
- ✅ Real-time injury impact
- ✅ Historical tournament performance
- ✅ Advanced efficiency metrics

The system is designed to identify value in a market that becomes increasingly inefficient during tournament play, when emotions run high and data-driven analysis provides the biggest edge.

**March Madness is where fortunes are made. Make sure you're on the right side of the data.**