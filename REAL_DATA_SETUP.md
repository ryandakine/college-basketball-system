# 🎯 REAL DATA SETUP GUIDE

## ⚠️ CRITICAL: This System ONLY Uses REAL Data

**NO synthetic data. NO mock data. NO simulated data.**

This guide shows you how to set up ALL real data sources for accurate predictions.

---

## 📋 Data Sources Overview

| Source | Type | Cost | Setup Time | Required? |
|--------|------|------|------------|-----------|
| **Barttorvik** | Team Efficiency | FREE | 5 min | ✅ REQUIRED |
| **ESPN API** | Injuries + Games | FREE | 2 min | ✅ REQUIRED |
| **The Odds API** | Real Betting Lines | FREE* | 5 min | ✅ REQUIRED |
| **KenPom** | Team Efficiency | $19.95/year | 10 min | ⭐ Optional |

*Free tier: 500 requests/month (enough for daily use)

---

## 🚀 Quick Setup (15 Minutes)

### Step 1: Get The Odds API Key (FREE)

**This gives you REAL betting lines from FanDuel, DraftKings, BetMGM, etc.**

1. Go to https://the-odds-api.com/
2. Click "Get Free API Key"
3. Sign up (email + password)
4. Copy your API key
5. Add to `.env` file:

```bash
ODDS_API_KEY=your_key_here
```

**Free Tier:**
- 500 requests/month
- Real odds from 15+ sportsbooks
- Covers ~16 games/day = 30 days

### Step 2: Test Barttorvik Scraper (FREE)

**FREE alternative to KenPom - same quality data!**

```bash
python barttorvik_scraper.py
```

This scrapes:
- Adjusted Offensive Efficiency (AdjOE)
- Adjusted Defensive Efficiency (AdjDE)
- Tempo
- Current season records
- All 350+ Division I teams

**No signup needed - completely free!**

### Step 3: Test ESPN Injury Fetcher (FREE)

```bash
python espn_injury_fetcher.py
```

Fetches current injury reports for all teams.

**No API key needed - uses public ESPN API!**

### Step 4: Run Complete Data Pipeline

```bash
python real_data_pipeline.py
```

This:
1. Fetches Barttorvik ratings
2. Fetches ESPN injuries
3. Fetches real betting odds
4. Validates ALL data is REAL (not synthetic)
5. Saves to database

**Expected output:**
```
✅ REAL DATA PIPELINE COMPLETE
   3/3 data sources successful
   ALL DATA IS REAL - NO SYNTHETIC DATA

📊 REAL DATA SUMMARY
==================
🎯 Barttorvik Ratings: 362 teams
🏥 Injury Reports: 45 total injuries
💰 Real Odds: 12 upcoming games

✅ ALL DATA IS REAL - NO SYNTHETIC DATA
```

---

## 🎯 OPTIONAL: KenPom Setup ($19.95/year)

**KenPom is slightly more accurate than Barttorvik but costs money.**

If you want to upgrade:

1. Subscribe at https://kenpom.com/
2. Get your username + password
3. Add to `.env`:

```bash
KENPOM_USERNAME=your_username
KENPOM_PASSWORD=your_password
```

4. Use `kenpom_scraper.py` (create this if you subscribe)

**Note:** Barttorvik is FREE and 95%+ as good as KenPom!

---

## 📊 Data Updates

### Daily (Automated)

```bash
# Add to cron (runs daily at 8 AM)
0 8 * * * cd /path/to/basketball-system && python real_data_pipeline.py
```

### Manual

```bash
# Update all data sources
python real_data_pipeline.py

# Update specific source
python barttorvik_scraper.py
python espn_injury_fetcher.py
python real_odds_fetcher.py
```

---

## ✅ Data Validation

The system **automatically validates** that all data is REAL:

### Validation Checks:

1. **No synthetic markers**
   - Rejects data containing: "mock", "fake", "synthetic", "dummy", "test"

2. **Realistic ranges**
   - Efficiency ratings: 80-130 range
   - Odds: -2000 to +2000 range

3. **Source verification**
   - Barttorvik: Checks data from barttorvik.com
   - ESPN: Verifies ESPN API responses
   - Odds: Confirms real sportsbook data

### If Validation Fails:

```
🚫 SYNTHETIC DATA DETECTED in barttorvik: 'mock' found
❌ REAL DATA PIPELINE FAILED
   CANNOT MAKE PREDICTIONS WITHOUT REAL DATA
```

**You CANNOT make predictions with synthetic data!**

---

## 🔍 Verify Your Data

```bash
# Check what data you have
python -c "
from real_data_pipeline import RealDataPipeline
pipeline = RealDataPipeline()
pipeline.print_data_summary()
"
```

Should show:
```
📊 REAL DATA SUMMARY
====================
🎯 Barttorvik Ratings: 362 teams
🏥 Injury Reports: 45 total injuries
💰 Real Odds: 12 upcoming games

✅ ALL DATA IS REAL - NO SYNTHETIC DATA
```

---

## 🎯 Example: Get Game Data

```python
from real_data_pipeline import RealDataPipeline

pipeline = RealDataPipeline()

# First, fetch all data
pipeline.fetch_all_real_data()

# Get data for specific game
game_data = pipeline.get_game_data("Duke", "North Carolina")

print(game_data)
```

Output:
```python
{
    'home_team': 'Duke',
    'away_team': 'North Carolina',
    'data_source': 'REAL',  # ← Guaranteed real!
    'home_adj_oe': 118.5,
    'home_adj_de': 95.2,
    'home_tempo': 71.3,
    'away_adj_oe': 115.8,
    'away_adj_de': 97.1,
    'away_tempo': 69.5,
    'home_injuries': 1,
    'away_injuries': 0,
    'home_odds': -150,
    'away_odds': +130,
    'spread': -3.5,
    'total': 152.5
}
```

**All numbers are REAL!**

---

## 🚫 What This System REJECTS

The data pipeline **automatically rejects**:

1. ❌ Mock data
2. ❌ Synthetic data
3. ❌ Simulated data
4. ❌ Test data
5. ❌ Dummy data
6. ❌ Generated data
7. ❌ Placeholder data

**If you try to use synthetic data, predictions will FAIL.**

---

## 💰 Cost Breakdown

**Total Monthly Cost: $0**

| Source | Monthly Cost |
|--------|--------------|
| Barttorvik | $0 |
| ESPN API | $0 |
| The Odds API | $0 (free tier) |
| **TOTAL** | **$0/month** |

**Optional:**
- KenPom: $1.66/month ($19.95/year)

---

## 📈 Data Quality

### Barttorvik vs KenPom

| Metric | Barttorvik (FREE) | KenPom ($20/yr) |
|--------|-------------------|-----------------|
| Accuracy | 95%+ | 100% (baseline) |
| Teams Covered | 350+ | 350+ |
| Updates | Daily | Daily |
| Historical Data | 10+ years | 20+ years |
| Cost | FREE | $19.95/year |

**Verdict:** Start with Barttorvik. Upgrade to KenPom if you need that extra 5%.

---

## 🔧 Troubleshooting

### "API key required!"

Add to `.env`:
```bash
ODDS_API_KEY=your_key_here
```

### "Could not find ratings table"

Barttorvik website structure changed. Check:
```bash
curl https://barttorvik.com/trank.php?year=2024
```

### "No upcoming games with odds"

Either:
1. No games scheduled today
2. API quota exceeded (500/month)
3. Check API key is valid

### "Insufficient data for backtest!"

You need historical data first:
```bash
python real_historical_data_scraper.py
```

---

## ✅ Final Checklist

Before making predictions, ensure:

- [ ] Odds API key in `.env`
- [ ] Ran `python real_data_pipeline.py` successfully
- [ ] Saw "ALL DATA IS REAL - NO SYNTHETIC DATA"
- [ ] Database has teams, injuries, and odds
- [ ] Data is less than 24 hours old

**If all checked, you're ready to make REAL predictions!**

---

## 🚀 Next Steps

1. **Setup data sources** (this guide)
2. **Fetch real data:** `python real_data_pipeline.py`
3. **Backtest system:** `./run_backtest.sh`
4. **Make predictions:** `python basketball_main.py --full-auto`

---

## 📞 Support

**Odds API Issues:**
- Docs: https://the-odds-api.com/liveapi/guides/v4/
- Support: contact@the-odds-api.com

**Barttorvik Issues:**
- Website: https://barttorvik.com/
- Check if website structure changed

**ESPN API Issues:**
- ESPN API may change - check:
  https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams

---

## 💪 Ready?

```bash
# Run this NOW
python real_data_pipeline.py
```

If successful, you have REAL data and can make predictions! 🎯
