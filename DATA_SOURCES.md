# Real Data Sources Configuration Guide

This document describes the real data sources required for the college basketball betting system to operate with live data.

## Overview

The system has been cleaned of all mock/synthetic data. It now requires proper API configuration to access real data sources. Without these configurations, the system will fail gracefully with clear error messages.

## Required Data Sources

### 1. Odds & Betting Lines

**API Provider**: The Odds API
- **Website**: https://the-odds-api.com
- **Sign-up**: Free tier available (limited requests), paid tiers for production use
- **Configuration File**: `monitor_config.json`
- **Configuration Key**: `api_keys.odds_api_key`
- **Usage**: Line monitoring, odds tracking, futures markets

**Setup Instructions**:

1. Create account at https://the-odds-api.com
2. Get your API key from the dashboard
3. Update `monitor_config.json`:
   ```json
   {
     "api_keys": {
       "odds_api_key": "YOUR_ACTUAL_API_KEY_HERE"
     }
   }
   ```

**Endpoints Used**:
- `/v4/sports/basketball_ncaab/outrights` - Tournament futures
- `/v4/sports/basketball_ncaab/odds` - Game lines, totals, moneylines

**Rate Limits**: 
- Free tier: 500 requests/month
- Check their website for current pricing and limits

### 2. Sports Data & Statistics

**Recommended Providers**:

#### Option A: ESPN API (Unofficial)
- **Access**: Free, unofficial API
- **Data**: Game scores, team stats, schedules
- **Reliability**: Medium (no official support)
- **Documentation**: Community-maintained

#### Option B: SportsDataIO
- **Website**: https://sportsdata.io
- **Access**: Paid service
- **Data**: Comprehensive college basketball data
- **Reliability**: High (official API)

#### Option C: The Blue Alliance / NCAA Stats
- **Access**: Official NCAA statistics
- **Data**: Official game logs, team stats
- **Reliability**: High
- **Note**: May require scraping or partnership

**Required Data**:
- Game logs (scores, possessions, efficiency metrics)
- Team statistics (tempo, shooting percentages, rebounds)
- Season statistics (conference records, rankings)
- Schedule data (dates, locations, opponents)

### 3. Injury Reports

**Recommended Providers**:

#### Option A: RotoBaller / RotoWire
- **Access**: Subscription required
- **Data**: Injury reports, player status updates
- **Update Frequency**: Real-time during season

#### Option B: Official Team Websites
- **Access**: Free but requires scraping
- **Data**: Official injury reports
- **Reliability**: High accuracy but manual updates needed

**Required Data**:
- Current injured players
- Injury severity and type
- Expected return dates
- Player statistics (PPG, RPG, APG, minutes)
- Depth chart positions

### 4. Player & Roster Data

**Recommended Providers**:

#### Option A: Sports Reference (College Basketball Reference)
- **Website**: https://www.sports-reference.com/cbb/
- **Access**: Free with rate limiting, paid for high-volume
- **Data**: Historical and current player stats, rosters

#### Option B: NCAA Official Sources
- **Access**: Varies by conference
- **Data**: Official rosters, player information

**Required Data**:
- Player profiles (position, height, weight, class year)
- Per-game statistics
- Advanced metrics (PER, efficiency ratings)
- Roster depth charts

## System Behavior Without APIs

### Line Monitor (`line_monitor.py`)
- **Without API key**: Returns empty odds data with error message
- **With invalid API key**: Logs error, returns empty data
- **Error handling**: Graceful failure, clear logging

### Analyzer Modules
- **Missing modules**: Falls back to MockAnalyzer with warnings
- **Logging**: Clear warnings indicate degraded mode
- **Functionality**: Returns neutral scores (0.5) when unavailable

### Data Collection Scripts
- **Without APIs**: Will fail with descriptive errors
- **Recommendation**: Configure APIs before running collection scripts

## Configuration Files

### `monitor_config.json`
Primary configuration for line monitoring:

```json
{
  "sportsbooks": ["draftkings", "fanduel", "betmgm", "caesars"],
  "check_interval": 300,
  "email_alerts": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "sender_password": "app_password",
    "recipient_email": "your_email@gmail.com"
  },
  "api_keys": {
    "odds_api_key": "your_odds_api_key_here",
    "sports_data_api_key": "configure_if_needed",
    "injury_api_key": "configure_if_needed"
  }
}
```

## Data Update Frequencies

| Data Type | Recommended Frequency | Critical Times |
|-----------|----------------------|----------------|
| Betting Lines | Every 5-10 minutes | 2-3 hours before games |
| Injury Reports | Hourly | Day of games |
| Game Stats | Post-game | Immediately after games |
| Player Stats | Daily | During active season |
| Rankings/Ratings | Daily | Weekly during season |

## Best Practices

1. **API Key Security**:
   - Never commit API keys to version control
   - Use environment variables when possible
   - Rotate keys periodically

2. **Rate Limiting**:
   - Respect API rate limits
   - Implement exponential backoff for failed requests
   - Cache data when appropriate

3. **Data Validation**:
   - Always validate API responses
   - Handle missing/null data gracefully
   - Log data quality issues

4. **Error Handling**:
   - System fails gracefully without APIs
   - Clear error messages for debugging
   - Logging for all API interactions

## Testing Without Production APIs

For development and testing:

1. **Use Free Tiers**: Most APIs offer free tiers with limited requests
2. **Test Functions**: Use the `main()` functions in modules (marked as test data)
3. **Unit Tests**: Create unit tests with fixture data (clearly marked)
4. **Integration Tests**: Test with small API request volumes

## Cost Estimation

### Minimal Setup (Development)
- Odds API: Free tier (500 requests/month)
- Sports Data: Free unofficial APIs
- **Total**: $0/month

### Production Setup
- Odds API: ~$50-100/month (depends on volume)
- Sports Data: ~$50-200/month
- Injury Data: ~$30-50/month
- **Total**: $130-350/month

## Support & Documentation

- **Odds API Docs**: https://the-odds-api.com/liveapi/guides/v4/
- **SportsDataIO Docs**: https://sportsdata.io/developers/api-documentation/ncaab
- **Sports Reference**: https://www.sports-reference.com/data_use.html

## Next Steps

1. ✅ Review required data sources
2. ⬜ Choose providers based on budget
3. ⬜ Sign up for API accounts
4. ⬜ Configure API keys in `monitor_config.json`
5. ⬜ Test API connections
6. ⬜ Implement data collection scripts
7. ⬜ Verify system functionality with real data
