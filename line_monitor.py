#!/usr/bin/env python3
"""
College Basketball Line Monitor
Tracks opening lines and alerts on value opportunities
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('line_monitor.log'),
        logging.StreamHandler()
    ]
)

class LineMonitor:
    def __init__(self, config_file: str = 'monitor_config.json'):
        """Initialize line monitor with configuration"""
        self.config = self.load_config(config_file)
        self.target_teams = self.load_target_teams()
        self.alert_thresholds = {
            'tournament_futures': 0.30,  # 30% edge minimum
            'season_totals': 0.15,       # 15% edge minimum
            'game_lines': 0.08           # 8% edge minimum
        }
        self.alerts_sent = {}  # Track sent alerts to avoid spam
        
    def load_config(self, config_file: str) -> Dict:
        """Load monitoring configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                "sportsbooks": ["draftkings", "fanduel", "betmgm", "caesars"],
                "check_interval": 300,  # 5 minutes
                "email_alerts": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "your_email@gmail.com",
                    "sender_password": "app_password",
                    "recipient_email": "your_email@gmail.com"
                },
                "api_keys": {
                    "odds_api_key": "your_odds_api_key_here"
                }
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logging.info(f"Created default config file: {config_file}")
            return default_config
    
    def load_target_teams(self) -> Dict:
        """Load target teams from preseason analysis"""
        try:
            with open('preseason_analysis_results.json', 'r') as f:
                data = json.load(f)
            
            # Extract high-value targets
            targets = {}
            for rec in data['recommendations']:
                team = rec['team'].lower().replace(' ', '_').replace("'", "")
                if rec['confidence'] in ['MASSIVE', 'STRONG']:
                    targets[team] = {
                        'name': rec['team'],
                        'bet_type': rec['bet_type'],
                        'confidence': rec['confidence'],
                        'expected_edge': rec.get('expected_edge', 0),
                        'projected_wins': rec.get('projected_wins'),
                        'market_total': rec.get('market_total'),
                        'reasoning': rec['reasoning']
                    }
            
            logging.info(f"Loaded {len(targets)} target teams for monitoring")
            return targets
            
        except FileNotFoundError:
            logging.error("Preseason analysis results not found. Run Monte Carlo analyzer first.")
            return {}
    
    def fetch_odds(self, sport: str = 'basketball_ncaab') -> Dict:
        """Fetch current odds from sports betting APIs"""
        api_key = self.config['api_keys'].get('odds_api_key')
        if not api_key or api_key == "your_odds_api_key_here":
            logging.error("No valid API key configured. Please configure 'odds_api_key' in monitor_config.json")
            logging.error("Get your API key from: https://the-odds-api.com")
            return {'futures': {}, 'games': {}, 'timestamp': datetime.now().isoformat(), 'error': 'No API key configured'}
        
        try:
            # Tournament futures
            futures_url = f"https://api.the-odds-api.com/v4/sports/{sport}/outrights"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'outrights'
            }
            
            response = requests.get(futures_url, params=params, timeout=10)
            response.raise_for_status()
            futures_data = response.json()
            
            # Season win totals and game lines
            games_url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'totals,h2h'
            }
            
            response = requests.get(games_url, params=params, timeout=10)
            response.raise_for_status()
            games_data = response.json()
            
            return {
                'futures': futures_data,
                'games': games_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            logging.error(f"API request failed: {e}")
            logging.error("Unable to fetch odds data. Please check your API key and internet connection.")
            return {'futures': {}, 'games': {}, 'timestamp': datetime.now().isoformat(), 'error': str(e)}
    

    
    def analyze_value_opportunities(self, odds_data: Dict) -> List[Dict]:
        """Analyze current odds for value opportunities"""
        opportunities = []
        
        # Check tournament futures
        for team_key, team_data in self.target_teams.items():
            if team_data['bet_type'] == 'Tournament Futures':
                market_key = f"{team_key}_tournament"
                if market_key in odds_data.get('futures', {}):
                    market_odds = odds_data['futures'][market_key]
                    market_prob = market_odds['implied_prob']
                    
                    # Get our projected probability from preseason analysis
                    our_prob = self.get_tournament_probability(team_data['name'])
                    
                    if our_prob and our_prob > market_prob:
                        edge = our_prob - market_prob
                        if edge >= self.alert_thresholds['tournament_futures']:
                            opportunities.append({
                                'type': 'Tournament Futures',
                                'team': team_data['name'],
                                'market_prob': market_prob,
                                'our_prob': our_prob,
                                'edge': edge,
                                'odds': market_odds['odds'],
                                'confidence': team_data['confidence'],
                                'priority': 'HIGH' if edge > 0.35 else 'MEDIUM'
                            })
        
        # Check season win totals
        for team_key, team_data in self.target_teams.items():
            if team_data['bet_type'] == 'Season Win Total':
                market_key = f"{team_key}_wins"
                if market_key in odds_data.get('season_totals', {}):
                    market_data = odds_data['season_totals'][market_key]
                    market_total = market_data['total']
                    projected_wins = team_data.get('projected_wins', 0)
                    
                    if projected_wins > market_total:
                        edge = (projected_wins - market_total) / market_total
                        if edge >= self.alert_thresholds['season_totals']:
                            opportunities.append({
                                'type': 'Season Win Total OVER',
                                'team': team_data['name'],
                                'market_total': market_total,
                                'projected_wins': projected_wins,
                                'edge': edge,
                                'over_odds': market_data['over_odds'],
                                'confidence': team_data['confidence'],
                                'priority': 'HIGH' if edge > 0.25 else 'MEDIUM'
                            })
        
        return opportunities
    
    def get_tournament_probability(self, team_name: str) -> Optional[float]:
        """Get tournament probability from preseason analysis"""
        try:
            with open('preseason_analysis_results.json', 'r') as f:
                data = json.load(f)
            
            team_key = team_name.lower().replace(' ', '_').replace("'", "")
            if team_key in data['teams']:
                return data['teams'][team_key]['tournament_probability']
            return None
        except (FileNotFoundError, KeyError):
            return None
    
    def send_alert(self, opportunities: List[Dict]) -> None:
        """Send email alert for value opportunities"""
        if not opportunities or not self.config['email_alerts']['enabled']:
            return
        
        # Check if we've already sent alerts for these opportunities
        alert_key = f"{datetime.now().strftime('%Y-%m-%d-%H')}"
        if alert_key in self.alerts_sent:
            return
        
        subject = f"🔥 COLLEGE BASKETBALL VALUE ALERT - {len(opportunities)} Opportunities"
        
        body = f"""
🏀 COLLEGE BASKETBALL BETTING ALERTS
===================================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Opportunities Found: {len(opportunities)}

HIGH PRIORITY BETS:
"""
        
        high_priority = [opp for opp in opportunities if opp['priority'] == 'HIGH']
        for opp in high_priority:
            if opp['type'] == 'Tournament Futures':
                body += f"""
🎯 {opp['team']} Tournament Futures
   Odds: {opp['odds']:+d} (Implied: {opp['market_prob']:.1%})
   Our Model: {opp['our_prob']:.1%}
   Edge: {opp['edge']:.1%}
   Confidence: {opp['confidence']}
"""
            elif opp['type'] == 'Season Win Total OVER':
                body += f"""
📊 {opp['team']} Season Win Total OVER {opp['market_total']}
   Over Odds: {opp['over_odds']:+d}
   Projected Wins: {opp['projected_wins']:.1f}
   Edge: {opp['edge']:.1%}
   Confidence: {opp['confidence']}
"""
        
        medium_priority = [opp for opp in opportunities if opp['priority'] == 'MEDIUM']
        if medium_priority:
            body += "\n\nMEDIUM PRIORITY BETS:\n"
            for opp in medium_priority:
                body += f"• {opp['team']} ({opp['type']}) - Edge: {opp['edge']:.1%}\n"
        
        body += f"""

⚠️  IMPORTANT REMINDERS:
• Early season lines are softest - ACT FAST!
• Maximum edge window: Next 20-25 days
• Recommended stakes: 3-5% for MASSIVE confidence, 2-4% for STRONG
• Monitor line movements throughout the day

🎯 This is your competitive advantage window!
"""
        
        try:
            self.send_email(subject, body)
            self.alerts_sent[alert_key] = datetime.now()
            logging.info(f"Alert sent for {len(opportunities)} opportunities")
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def send_email(self, subject: str, body: str) -> None:
        """Send email notification"""
        email_config = self.config['email_alerts']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
    
    def log_opportunities(self, opportunities: List[Dict]) -> None:
        """Log opportunities to file"""
        if not opportunities:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'opportunities_count': len(opportunities),
            'opportunities': opportunities
        }
        
        # Append to opportunities log
        try:
            with open('opportunities_log.json', 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            log_data = []
        
        log_data.append(log_entry)
        
        # Keep only last 1000 entries
        if len(log_data) > 1000:
            log_data = log_data[-1000:]
        
        with open('opportunities_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def run_monitoring_cycle(self) -> None:
        """Run one monitoring cycle"""
        logging.info("🔍 Starting monitoring cycle...")
        
        # Fetch current odds
        odds_data = self.fetch_odds()
        
        # Analyze for opportunities
        opportunities = self.analyze_value_opportunities(odds_data)
        
        if opportunities:
            logging.info(f"🎯 Found {len(opportunities)} value opportunities")
            
            # Send alerts
            self.send_alert(opportunities)
            
            # Log opportunities
            self.log_opportunities(opportunities)
            
            # Print to console
            print(f"\n🔥 VALUE OPPORTUNITIES DETECTED ({len(opportunities)}):")
            print("=" * 50)
            for opp in opportunities:
                print(f"• {opp['team']} - {opp['type']} - Edge: {opp['edge']:.1%} ({opp['priority']})")
        else:
            logging.info("No new value opportunities detected")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        logging.info("🚀 Starting College Basketball Line Monitor")
        logging.info(f"📊 Monitoring {len(self.target_teams)} target teams")
        logging.info(f"⏰ Check interval: {self.config['check_interval']} seconds")
        
        try:
            while True:
                self.run_monitoring_cycle()
                
                # Wait for next cycle
                time.sleep(self.config['check_interval'])
                
        except KeyboardInterrupt:
            logging.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logging.error(f"❌ Monitoring error: {e}")
            raise

def main():
    """Main function"""
    print("🏀 COLLEGE BASKETBALL LINE MONITOR")
    print("=" * 40)
    
    monitor = LineMonitor()
    
    # Check if we have target teams
    if not monitor.target_teams:
        print("❌ No target teams found. Run preseason Monte Carlo analyzer first.")
        return
    
    print(f"📊 Monitoring {len(monitor.target_teams)} high-value targets")
    print(f"⚠️  Configure email alerts in monitor_config.json")
    print(f"🎯 Soft line window: ~20-25 days remaining!")
    print("-" * 40)
    
    # Run one cycle to test
    print("🔍 Running initial scan...")
    monitor.run_monitoring_cycle()
    
    # Ask user if they want continuous monitoring
    response = input("\n🚀 Start continuous monitoring? (y/N): ")
    if response.lower() in ['y', 'yes']:
        print("🏃‍♂️ Starting continuous monitoring... (Ctrl+C to stop)")
        monitor.start_monitoring()
    else:
        print("✅ Single scan complete. Check opportunities_log.json for details.")

if __name__ == "__main__":
    main()