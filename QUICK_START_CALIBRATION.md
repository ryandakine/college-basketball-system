# Quick Start: Calibrated Prediction System

## Installation

Install required dependency:
```bash
pip install filelock --break-system-packages
```

## Making a Prediction with Calibration

```python
from core_basketball_prediction_engine import CoreBasketballPredictionEngine

# Initialize engine (automatically loads prediction logger and RAG system)
engine = CoreBasketballPredictionEngine()

# Prepare game data
game_data = {
    'game_id': 'duke_unc_20251115',
    'date': '2025-11-15',
    'home_team': 'Duke',
    'away_team': 'North Carolina',
    'home_conference': 'ACC',
    'away_conference': 'ACC',
    'home_kenpom_rating': 85.2,
    'away_kenpom_rating': 78.6,
    'home_team_stats': {
        'offensive_efficiency': 118.5,
        'defensive_efficiency': 95.2,
        'tempo': 72.3
    },
    'away_team_stats': {
        'offensive_efficiency': 115.8,
        'defensive_efficiency': 98.1,
        'tempo': 69.8
    },
    'venue': 'Cameron Indoor Stadium',
    'neutral_site': False,
    'betting_lines': {
        'spread': {
            'line': -3.5,
            'home_odds': -110,
            'away_odds': -110
        },
        'total': {
            'line': 148.5,
            'over_odds': -110,
            'under_odds': -110
        }
    },
    'odds': {
        'moneyline': {
            'home': -140,
            'away': +120
        }
    }
}

# Generate prediction
prediction = engine.generate_comprehensive_prediction(game_data)

# View results
print(f"Game: {prediction.game_context.home_team} vs {prediction.game_context.away_team}")
print(f"Win Probability: {prediction.final_win_probability:.1%}")
print(f"Confidence Tier: {prediction.confidence_tier}")

# Check betting recommendations
if prediction.spread_recommendation:
    rec = prediction.spread_recommendation
    print(f"\nSpread Recommendation: {rec.recommendation}")
    print(f"Raw Confidence: {rec.confidence:.1%}")  # Already calibrated
    print(f"Edge: {rec.edge_percentage:.1%}")
    print(f"Suggested Units: {rec.suggested_unit_size}")
    
    # Check for warnings
    for risk in rec.risk_factors:
        if "Problematic" in risk:
            print(f"⚠️  WARNING: {risk}")
```

## Checking Prediction Log

```python
from prediction_logger import PredictionLogger

logger = PredictionLogger()

# Get all predictions
predictions = logger.get_predictions()
print(f"Total predictions logged: {len(predictions)}")

# Get performance stats
stats = logger.get_performance_stats()
print(f"\nOverall Win Rate: {stats['win_rate']:.1%}")
print(f"ROI: {stats['roi']:.1f}%")
print(f"\nBy Confidence:")
print(f"  HIGH: {stats['by_confidence']['HIGH']['win_rate']:.1%} ({stats['by_confidence']['HIGH']['count']} bets)")
print(f"  MEDIUM: {stats['by_confidence']['MEDIUM']['win_rate']:.1%} ({stats['by_confidence']['MEDIUM']['count']} bets)")
print(f"  LOW: {stats['by_confidence']['LOW']['win_rate']:.1%} ({stats['by_confidence']['LOW']['count']} bets)")
```

## Using RAG Context

```python
from basketball_rag_system import BasketballRAGSystem

rag = BasketballRAGSystem()

# Get comprehensive context for a game
context = rag.get_comprehensive_context(
    team_a="Duke",
    team_b="North Carolina",
    venue="Cameron Indoor Stadium",
    referee="John Higgins"  # If known
)

# View insights
print("\n🔍 Contextual Insights:")
for insight in context['contextual_insights']:
    print(f"   {insight}")

# Get specific referee info
if context['referee_profile']:
    ref = context['referee_profile']
    print(f"\n🔵 Referee: {ref['referee_name']}")
    print(f"   Pace Factor: {ref['pace_factor']:.2f}x")
    print(f"   Home Bias: {ref['home_foul_bias']:+.1f}")

# Get historical matchup
if context['historical_matchup']:
    matchup = context['historical_matchup']
    print(f"\n📊 Historical: {matchup['team_a_wins']}-{matchup['team_b_wins']}")
    print(f"   Avg Total: {matchup['avg_total_points']:.1f}")
```

## Understanding Calibration

### Before Calibration
```
Raw Confidence: 87%
Issue: Model might not actually win 87% of these bets
```

### After Calibration (80%+ predictions)
```
Raw Confidence: 87%
Calibrated: 74% (87% × 0.85)
Result: More realistic confidence that matches actual win rate
```

### Problematic Bucket (50-60%)
```
Confidence: 54%
Warning: ⚠️ Problematic 50-60% confidence bucket
Action: Consider skipping bet or retraining model
```

## Best Practices

### 1. Start Logging Immediately
Every prediction is automatically logged when using the engine. After 50-100 predictions, analyze:
```python
stats = logger.get_performance_stats()
```

### 2. Respect Warnings
If you see the 50-60% confidence warning, seriously consider:
- Skipping the bet
- Reducing unit size to minimum
- Waiting for more data/better matchup

### 3. Use RAG Context
Before betting close games (50-60%), check RAG insights:
```python
context = rag.get_comprehensive_context(team_a, team_b, venue, referee)
```

### 4. Track Performance by Tier
Focus on HIGH confidence bets until you have enough data:
```python
high_conf_bets = logger.get_predictions({'recommendation': 'STRONG_BET'})
```

### 5. Update Outcomes
After each game, update the prediction log:
```python
logger.update_outcome(prediction_id, {
    'actual_result': 'Duke won 78-75',
    'bet_outcome': 'WIN',  # or 'LOSS' or 'PUSH'
    'profit_loss': 0.91  # Units won/lost
})
```

## Adjusting Calibration

If after 100+ predictions you find calibration needs adjustment:

```python
# In core_basketball_prediction_engine.py

# Make calibration more aggressive (multiply by 0.80 instead of 0.85)
self.CALIBRATION_MULTIPLIER = 0.80

# Change the threshold (apply to 75%+ instead of 80%+)
self.CALIBRATION_THRESHOLD = 0.75

# Adjust problematic bucket range
self.PROBLEMATIC_CONFIDENCE_MIN = 0.48
self.PROBLEMATIC_CONFIDENCE_MAX = 0.62
```

## Expected Outcomes

After implementing these improvements, you should see:

✅ **Better Calibration**
- 75% confidence predictions win ~75% of the time
- No more false confidence on close games

✅ **Improved Decision Making**
- Clear warnings on problematic predictions
- Rich context from RAG system
- Data-driven confidence tiers

✅ **Complete Tracking**
- Every prediction logged
- Performance by confidence tier
- ROI by bet type

✅ **Contextual Intelligence**
- Referee tendencies inform predictions
- Historical matchups provide context
- Venue-specific adjustments

## Support

See `CALIBRATION_IMPROVEMENTS.md` for detailed documentation.

For issues or questions:
- Check prediction logs: `data/predictions/prediction_log.json`
- Review RAG database: `data/basketball_rag.db`
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
