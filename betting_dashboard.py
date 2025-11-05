#!/usr/bin/env python3
"""
College Basketball Master Betting Dashboard
Combines all systems and provides unified interface for betting decisions
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BettingDashboard:
    def __init__(self):
        """Initialize the master betting dashboard"""
        self.systems = {
            'preseason_monte_carlo': 'preseason_monte_carlo_analyzer.py',
            'line_monitor': 'line_monitor.py', 
            'early_season_analyzer': 'early_season_analyzer.py',
            'integrated_system': 'integrated_betting_system.py'
        }
        
        self.data_files = {
            'preseason_analysis': 'preseason_analysis_results.json',
            'early_season_analysis': 'early_season_analysis.json',
            'opportunities_log': 'opportunities_log.json',
            'line_monitor_log': 'line_monitor.log'
        }
        
    def check_system_status(self) -> Dict:
        """Check status of all betting systems"""
        status = {}
        
        # Check if system files exist
        for system_name, filename in self.systems.items():
            status[system_name] = {
                'file_exists': os.path.exists(filename),
                'executable': os.access(filename, os.X_OK) if os.path.exists(filename) else False
            }
        
        # Check if data files exist and are recent
        for data_name, filename in self.data_files.items():
            if os.path.exists(filename):
                mtime = os.path.getmtime(filename)
                modified_time = datetime.fromtimestamp(mtime)
                age_hours = (datetime.now() - modified_time).total_seconds() / 3600
                
                status[data_name] = {
                    'exists': True,
                    'modified': modified_time.isoformat(),
                    'age_hours': round(age_hours, 1),
                    'fresh': age_hours < 24  # Consider fresh if less than 24 hours old
                }
            else:
                status[data_name] = {
                    'exists': False,
                    'modified': None,
                    'age_hours': None,
                    'fresh': False
                }
        
        return status
    
    def get_master_recommendations(self) -> Dict:
        """Get unified betting recommendations from all systems"""
        recommendations = {
            'tournament_futures': [],
            'season_totals': [],
            'game_spreads': [],
            'game_totals': [],
            'high_priority': [],
            'medium_priority': [],
            'alerts': []
        }
        
        # Load preseason Monte Carlo recommendations
        try:
            with open('preseason_analysis_results.json', 'r') as f:
                preseason_data = json.load(f)
            
            for rec in preseason_data.get('recommendations', []):
                if rec['bet_type'] == 'Tournament Futures':
                    recommendations['tournament_futures'].append({
                        'source': 'Monte Carlo',
                        'team': rec['team'],
                        'confidence': rec['confidence'],
                        'edge': rec['expected_edge'],
                        'reasoning': rec['reasoning'],
                        'stake': rec['stake_recommendation']
                    })
                elif rec['bet_type'] == 'Season Win Total':
                    recommendations['season_totals'].append({
                        'source': 'Monte Carlo',
                        'team': rec['team'],
                        'recommendation': rec['recommendation'],
                        'confidence': rec['confidence'],
                        'projected_wins': rec.get('projected_wins', 0),
                        'market_total': rec.get('market_total', 0),
                        'edge': rec.get('edge', 0),
                        'stake': rec['stake_recommendation']
                    })
                
                # Categorize by priority
                if rec['confidence'] == 'MASSIVE':
                    recommendations['high_priority'].append({
                        'type': rec['bet_type'],
                        'team': rec['team'],
                        'confidence': rec['confidence'],
                        'source': 'Monte Carlo'
                    })
                
        except FileNotFoundError:
            logging.warning("Preseason analysis not found")
        
        # Load early season game recommendations
        try:
            with open('early_season_analysis.json', 'r') as f:
                early_season_data = json.load(f)
            
            for opp in early_season_data.get('opportunities', []):
                if opp['type'] == 'Spread':
                    recommendations['game_spreads'].append({
                        'source': 'Early Season',
                        'game': opp['game'],
                        'recommendation': opp['recommendation'],
                        'confidence': opp['confidence'],
                        'edge': opp['edge'],
                        'reasoning': opp['reasoning'],
                        'date': opp['date']
                    })
                elif opp['type'] == 'Total':
                    recommendations['game_totals'].append({
                        'source': 'Early Season',
                        'game': opp['game'],
                        'recommendation': opp['recommendation'],
                        'confidence': opp['confidence'],
                        'edge': opp['edge'],
                        'reasoning': opp['reasoning'],
                        'date': opp['date']
                    })
                
                # Categorize by priority
                if opp['confidence'] == 'HIGH':
                    recommendations['high_priority'].append({
                        'type': opp['type'],
                        'game': opp['game'],
                        'confidence': opp['confidence'],
                        'source': 'Early Season',
                        'date': opp['date']
                    })
                elif opp['confidence'] == 'MEDIUM':
                    recommendations['medium_priority'].append({
                        'type': opp['type'],
                        'game': opp['game'],
                        'confidence': opp['confidence'],
                        'source': 'Early Season',
                        'date': opp['date']
                    })
            
        except FileNotFoundError:
            logging.warning("Early season analysis not found")
        
        # Load recent alerts from line monitor
        try:
            with open('opportunities_log.json', 'r') as f:
                opportunities_log = json.load(f)
            
            # Get alerts from last 24 hours
            recent_alerts = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for log_entry in opportunities_log[-10:]:  # Check last 10 entries
                log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                if log_time.replace(tzinfo=None) > cutoff_time:
                    recent_alerts.extend(log_entry['opportunities'])
            
            recommendations['alerts'] = recent_alerts[-5:]  # Keep last 5 alerts
            
        except FileNotFoundError:
            logging.warning("Opportunities log not found")
        
        return recommendations
    
    def calculate_portfolio_allocation(self, recommendations: Dict) -> Dict:
        """Calculate recommended portfolio allocation across all opportunities"""
        allocations = {
            'massive_confidence': 0,    # 3-5% each
            'strong_confidence': 0,     # 2-4% each  
            'medium_confidence': 0,     # 1-2% each
            'total_allocation': 0,
            'max_recommended': 0.25     # Max 25% of bankroll
        }
        
        # Count opportunities by confidence level
        massive_count = len([r for r in recommendations['tournament_futures'] if r.get('confidence') == 'MASSIVE'])
        strong_count = len([r for r in recommendations['season_totals'] if r.get('confidence') == 'STRONG'])
        medium_count = len([r for r in recommendations['game_spreads'] + recommendations['game_totals'] 
                           if r.get('confidence') == 'MEDIUM'])
        
        # Calculate allocations
        allocations['massive_confidence'] = massive_count * 0.04  # 4% average
        allocations['strong_confidence'] = strong_count * 0.03    # 3% average
        allocations['medium_confidence'] = medium_count * 0.015   # 1.5% average
        
        allocations['total_allocation'] = (allocations['massive_confidence'] + 
                                         allocations['strong_confidence'] + 
                                         allocations['medium_confidence'])
        
        return allocations
    
    def run_system_update(self, system_name: str) -> bool:
        """Run a specific system to update its data"""
        if system_name not in self.systems:
            logging.error(f"Unknown system: {system_name}")
            return False
        
        script_path = self.systems[system_name]
        if not os.path.exists(script_path):
            logging.error(f"System script not found: {script_path}")
            return False
        
        try:
            logging.info(f"Running {system_name}...")
            result = subprocess.run(['python3', script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logging.info(f"✅ {system_name} completed successfully")
                return True
            else:
                logging.error(f"❌ {system_name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"❌ {system_name} timed out")
            return False
        except Exception as e:
            logging.error(f"❌ Error running {system_name}: {e}")
            return False
    
    def generate_daily_report(self) -> str:
        """Generate comprehensive daily betting report"""
        status = self.check_system_status()
        recommendations = self.get_master_recommendations()
        allocations = self.calculate_portfolio_allocation(recommendations)
        
        report = f"""
🏀 COLLEGE BASKETBALL BETTING DASHBOARD
{'='*50}
📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏰ Days Until Season: ~20-25 (CRITICAL SOFT LINE WINDOW!)

📊 SYSTEM STATUS:
{'-'*30}
"""
        
        for system, info in status.items():
            if 'file_exists' in info:
                status_emoji = "✅" if info['file_exists'] else "❌"
                report += f"{status_emoji} {system.replace('_', ' ').title()}: {'Active' if info['file_exists'] else 'Missing'}\n"
            elif 'exists' in info:
                if info['exists']:
                    freshness = "🟢 Fresh" if info['fresh'] else "🟡 Stale" if info['age_hours'] < 168 else "🔴 Old"
                    report += f"  📄 {system.replace('_', ' ').title()}: {freshness} ({info['age_hours']:.1f}h old)\n"
                else:
                    report += f"  📄 {system.replace('_', ' ').title()}: ❌ Missing\n"
        
        # High Priority Opportunities
        report += f"\n🔥 HIGH PRIORITY OPPORTUNITIES:\n{'-'*35}\n"
        high_priority = recommendations['high_priority'][:5]  # Top 5
        if high_priority:
            for opp in high_priority:
                report += f"⭐ {opp.get('team', opp.get('game', 'Unknown'))} - {opp['type']} ({opp['source']})\n"
                if 'date' in opp:
                    report += f"   📅 {opp['date']}\n"
        else:
            report += "No high priority opportunities detected.\n"
        
        # Tournament Futures
        if recommendations['tournament_futures']:
            report += f"\n🎯 TOURNAMENT FUTURES ({len(recommendations['tournament_futures'])}):\n"
            for fut in recommendations['tournament_futures'][:5]:
                report += f"• {fut['team']}: {fut['confidence']} confidence, {fut['edge']:.1%} edge\n"
        
        # Season Win Totals  
        if recommendations['season_totals']:
            report += f"\n📊 SEASON WIN TOTALS ({len(recommendations['season_totals'])}):\n"
            for total in recommendations['season_totals'][:5]:
                report += f"• {total['team']} {total['recommendation']}: {total['confidence']} confidence\n"
        
        # Early Season Games
        if recommendations['game_spreads'] or recommendations['game_totals']:
            game_count = len(recommendations['game_spreads']) + len(recommendations['game_totals'])
            report += f"\n🏀 EARLY SEASON GAMES ({game_count}):\n"
            
            all_games = recommendations['game_spreads'] + recommendations['game_totals']
            for game in sorted(all_games, key=lambda x: x.get('date', ''), reverse=True)[:5]:
                report += f"• {game['game']}: {game['recommendation']} ({game['confidence']})\n"
        
        # Portfolio Allocation
        report += f"\n💰 PORTFOLIO ALLOCATION:\n{'-'*25}\n"
        report += f"🔥 Massive Confidence Bets: {allocations['massive_confidence']:.1%} of bankroll\n"
        report += f"💪 Strong Confidence Bets: {allocations['strong_confidence']:.1%} of bankroll\n"
        report += f"📊 Medium Confidence Bets: {allocations['medium_confidence']:.1%} of bankroll\n"
        report += f"📈 Total Recommended: {allocations['total_allocation']:.1%} of bankroll\n"
        
        if allocations['total_allocation'] > allocations['max_recommended']:
            report += f"⚠️  WARNING: Total allocation exceeds recommended maximum ({allocations['max_recommended']:.1%})\n"
        
        # Recent Alerts
        if recommendations['alerts']:
            report += f"\n🚨 RECENT ALERTS ({len(recommendations['alerts'])}):\n"
            for alert in recommendations['alerts'][-3:]:
                report += f"• {alert.get('team', 'Unknown')}: {alert.get('type', 'Unknown')} - {alert.get('edge', 0):.1%} edge\n"
        
        # Action Items
        report += f"\n📋 ACTION ITEMS:\n{'-'*20}\n"
        
        stale_data = [name for name, info in status.items() 
                     if 'fresh' in info and not info['fresh']]
        if stale_data:
            report += f"🔄 Update stale data: {', '.join(stale_data)}\n"
        
        if not recommendations['tournament_futures']:
            report += f"🎯 Run preseason Monte Carlo analysis\n"
        
        if not (recommendations['game_spreads'] or recommendations['game_totals']):
            report += f"🏀 Run early season game analysis\n"
        
        if len(recommendations['alerts']) == 0:
            report += f"👀 Start line monitoring for real-time opportunities\n"
        
        report += f"\n⚠️  CRITICAL REMINDERS:\n"
        report += f"• 🕐 Soft line window closes in ~20-25 days!\n"
        report += f"• 💪 This is maximum edge opportunity period\n"
        report += f"• 📱 Monitor line movements throughout the day\n"
        report += f"• 🎯 Focus on tournament futures and season totals\n"
        report += f"• 💰 Stick to bankroll management rules\n"
        
        report += f"\n🚀 READY TO DOMINATE THE EARLY SEASON MARKETS!"
        
        return report
    
    def save_daily_report(self, report: str) -> None:
        """Save daily report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"daily_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logging.info(f"Daily report saved to {filename}")
    
    def interactive_menu(self) -> None:
        """Interactive menu for system management"""
        while True:
            print("\n🏀 COLLEGE BASKETBALL BETTING DASHBOARD")
            print("=" * 45)
            print("1. 📊 Generate Daily Report")
            print("2. 🔄 Update Preseason Analysis")
            print("3. 🏀 Update Early Season Analysis")
            print("4. 👀 Start Line Monitoring")
            print("5. 📈 Run Integrated System")
            print("6. 📋 System Status Check")
            print("7. 💾 Export All Data")
            print("0. 🚪 Exit")
            
            try:
                choice = input("\n🎯 Select option: ").strip()
                
                if choice == '0':
                    print("👋 Good luck with your bets!")
                    break
                elif choice == '1':
                    report = self.generate_daily_report()
                    print(report)
                    
                    save = input("\n💾 Save report to file? (y/N): ").lower()
                    if save in ['y', 'yes']:
                        self.save_daily_report(report)
                
                elif choice == '2':
                    self.run_system_update('preseason_monte_carlo')
                
                elif choice == '3':
                    self.run_system_update('early_season_analyzer')
                
                elif choice == '4':
                    print("🚀 Starting line monitor...")
                    os.system('python3 line_monitor.py')
                
                elif choice == '5':
                    self.run_system_update('integrated_system')
                
                elif choice == '6':
                    status = self.check_system_status()
                    print("\n📊 SYSTEM STATUS:")
                    print("-" * 25)
                    for system, info in status.items():
                        if 'file_exists' in info:
                            status_str = "✅ Active" if info['file_exists'] else "❌ Missing"
                        else:
                            if info['exists']:
                                status_str = f"🟢 Fresh ({info['age_hours']:.1f}h)" if info['fresh'] else f"🟡 Stale ({info['age_hours']:.1f}h)"
                            else:
                                status_str = "❌ Missing"
                        print(f"{system.replace('_', ' ').title()}: {status_str}")
                
                elif choice == '7':
                    print("💾 Exporting all data...")
                    self.export_all_data()
                
                else:
                    print("❌ Invalid option. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def export_all_data(self) -> None:
        """Export all system data to a single archive"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        export_dir = f"betting_data_export_{timestamp}"
        
        try:
            os.makedirs(export_dir, exist_ok=True)
            
            files_to_export = list(self.data_files.values()) + ['monitor_config.json']
            
            for filename in files_to_export:
                if os.path.exists(filename):
                    os.system(f"cp {filename} {export_dir}/")
            
            # Create export summary
            summary = f"""
College Basketball Betting Data Export
Generated: {datetime.now().isoformat()}

Files included:
"""
            for filename in files_to_export:
                if os.path.exists(filename):
                    summary += f"✅ {filename}\n"
                else:
                    summary += f"❌ {filename} (not found)\n"
            
            with open(f"{export_dir}/export_summary.txt", 'w') as f:
                f.write(summary)
            
            print(f"✅ Data exported to {export_dir}/")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

def main():
    """Main function"""
    dashboard = BettingDashboard()
    
    # Generate initial report
    print("🏀 INITIALIZING BETTING DASHBOARD...")
    print("=" * 40)
    
    report = dashboard.generate_daily_report()
    print(report)
    
    # Start interactive menu
    dashboard.interactive_menu()

if __name__ == "__main__":
    main()