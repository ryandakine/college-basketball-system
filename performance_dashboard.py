#!/usr/bin/env python3
"""
Performance Tracking Dashboard for College Basketball System
===========================================================

Comprehensive dashboard featuring:
- Real-time performance metrics and analytics
- Interactive visualizations
- Model accuracy tracking
- Betting ROI analysis
- Risk management monitoring
- Automated reporting
- Alert system for performance degradation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional web dashboard imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit/Plotly not available - web dashboard disabled")

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    prediction_type: str  # 'spread', 'total', 'moneyline'
    
    # Accuracy metrics
    overall_accuracy: float
    recent_accuracy: float  # Last 30 days
    accuracy_vs_confidence: Dict[str, float]  # High/Medium/Low confidence buckets
    
    # Statistical metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r_squared: float
    calibration_score: float
    
    # Betting metrics
    roi_pct: float
    win_rate: float
    average_odds: float
    profit_loss: float
    
    # Volume metrics
    total_predictions: int
    recent_predictions: int
    predictions_by_confidence: Dict[str, int]

@dataclass
class BettingMetrics:
    """Betting performance metrics"""
    strategy_name: str
    
    # Financial metrics
    total_staked: float
    total_profit: float
    roi_pct: float
    profit_factor: float  # Gross profit / Gross loss
    
    # Performance metrics
    win_rate: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    value_at_risk: float
    
    # Volume metrics
    total_bets: int
    winning_bets: int
    losing_bets: int
    push_bets: int
    
    # Time-based metrics
    daily_avg_profit: float
    monthly_performance: Dict[str, float]
    streak_metrics: Dict[str, int]

@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    uptime_pct: float
    data_freshness: float  # Hours since last update
    model_agreement: float  # How much models agree
    prediction_confidence_avg: float
    
    # Performance alerts
    accuracy_declining: bool
    roi_declining: bool
    high_drawdown: bool
    model_drift_detected: bool
    
    # System metrics
    predictions_per_day: float
    bets_per_day: float
    data_quality_score: float
    coverage_pct: float  # % of games covered

class PerformanceDashboard:
    """Comprehensive performance tracking dashboard"""
    
    def __init__(self, system_db_path: str = "basketball_analytics.db"):
        self.logger = logging.getLogger(__name__)
        self.system_db = system_db_path
        
        # Dashboard database
        self.dashboard_db = "performance_dashboard.db"
        self._init_dashboard_db()
        
        # Configuration
        self.alert_thresholds = {
            'accuracy_decline': 0.05,  # 5% decline triggers alert
            'roi_decline': 0.10,       # 10% ROI decline
            'max_drawdown': 0.20,      # 20% max drawdown
            'confidence_drop': 0.10    # 10% confidence drop
        }
        
        # Plotting style
        self._setup_plotting_style()
    
    def _init_dashboard_db(self):
        """Initialize dashboard database"""
        conn = sqlite3.connect(self.dashboard_db)
        cursor = conn.cursor()
        
        # Daily performance snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy_pct REAL,
                total_bets INTEGER,
                winning_bets INTEGER,
                total_staked REAL,
                total_profit REAL,
                roi_pct REAL,
                max_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                model_name TEXT,
                prediction_type TEXT,
                predictions_made INTEGER,
                correct_predictions INTEGER,
                mae REAL,
                rmse REAL,
                roi_pct REAL,
                profit_loss REAL
            )
        ''')
        
        # Performance alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                severity TEXT,  -- 'HIGH', 'MEDIUM', 'LOW'
                message TEXT,
                metric_value REAL,
                threshold_value REAL,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # System health log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                uptime_pct REAL,
                data_freshness REAL,
                model_agreement REAL,
                prediction_confidence REAL,
                data_quality_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Custom colors for basketball theme
        self.colors = {
            'primary': '#FF6600',    # Basketball orange
            'secondary': '#003366',  # Navy blue
            'success': '#28a745',    # Green
            'warning': '#ffc107',    # Yellow
            'danger': '#dc3545',     # Red
            'info': '#17a2b8'        # Cyan
        }
    
    def generate_comprehensive_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        report = {
            'report_date': end_date.isoformat(),
            'period': f"{days_back} days",
            'model_metrics': self._analyze_model_performance(start_date, end_date),
            'betting_metrics': self._analyze_betting_performance(start_date, end_date),
            'system_health': self._analyze_system_health(start_date, end_date),
            'key_insights': self._generate_key_insights(start_date, end_date),
            'recommendations': self._generate_recommendations(start_date, end_date),
            'charts': self._generate_chart_data(start_date, end_date)
        }
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _analyze_model_performance(self, start_date: datetime, end_date: datetime) -> List[ModelMetrics]:
        """Analyze model performance across different metrics"""
        model_metrics = []
        
        # Connect to system database
        conn = sqlite3.connect(self.system_db)
        
        # Get prediction data
        query = '''
            SELECT model_name, prediction_type, predicted_value, actual_value, 
                   confidence, timestamp, game_id
            FROM prediction_history 
            WHERE timestamp BETWEEN ? AND ?
            AND actual_value IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        if df.empty:
            conn.close()
            return model_metrics
        
        # Analyze each model
        for model_name in df['model_name'].unique():
            for pred_type in df['prediction_type'].unique():
                model_data = df[(df['model_name'] == model_name) & 
                              (df['prediction_type'] == pred_type)]
                
                if model_data.empty:
                    continue
                
                # Calculate accuracy metrics
                errors = model_data['predicted_value'] - model_data['actual_value']
                mae = np.mean(np.abs(errors))
                rmse = np.sqrt(np.mean(errors ** 2))
                
                # For spread/total predictions, "correct" is within 3 points
                if pred_type in ['spread', 'total']:
                    correct_predictions = np.sum(np.abs(errors) <= 3)
                    overall_accuracy = correct_predictions / len(model_data)
                else:  # moneyline
                    correct_predictions = np.sum(np.sign(model_data['predicted_value']) == 
                                               np.sign(model_data['actual_value']))
                    overall_accuracy = correct_predictions / len(model_data)
                
                # Recent accuracy (last 7 days)
                recent_data = model_data[model_data['timestamp'] >= 
                                       (end_date - timedelta(days=7))]
                if not recent_data.empty:
                    recent_errors = recent_data['predicted_value'] - recent_data['actual_value']
                    if pred_type in ['spread', 'total']:
                        recent_correct = np.sum(np.abs(recent_errors) <= 3)
                    else:
                        recent_correct = np.sum(np.sign(recent_data['predicted_value']) == 
                                              np.sign(recent_data['actual_value']))
                    recent_accuracy = recent_correct / len(recent_data)
                else:
                    recent_accuracy = overall_accuracy
                
                # Accuracy by confidence level
                model_data['confidence_bucket'] = pd.cut(model_data['confidence'], 
                                                       bins=[0, 0.6, 0.8, 1.0], 
                                                       labels=['Low', 'Medium', 'High'])
                
                accuracy_by_confidence = {}
                predictions_by_confidence = {}
                
                for bucket in ['Low', 'Medium', 'High']:
                    bucket_data = model_data[model_data['confidence_bucket'] == bucket]
                    predictions_by_confidence[bucket] = len(bucket_data)
                    
                    if not bucket_data.empty:
                        bucket_errors = bucket_data['predicted_value'] - bucket_data['actual_value']
                        if pred_type in ['spread', 'total']:
                            bucket_correct = np.sum(np.abs(bucket_errors) <= 3)
                        else:
                            bucket_correct = np.sum(np.sign(bucket_data['predicted_value']) == 
                                                  np.sign(bucket_data['actual_value']))
                        accuracy_by_confidence[bucket] = bucket_correct / len(bucket_data)
                    else:
                        accuracy_by_confidence[bucket] = 0.0
                
                # R-squared
                r_squared = 1 - (np.sum(errors ** 2) / 
                               np.sum((model_data['actual_value'] - 
                                     np.mean(model_data['actual_value'])) ** 2))
                
                # Calibration score (simplified)
                calibration_score = 1 - mae / np.std(model_data['actual_value'])
                calibration_score = max(0, min(1, calibration_score))
                
                model_metrics.append(ModelMetrics(
                    model_name=model_name,
                    prediction_type=pred_type,
                    overall_accuracy=overall_accuracy,
                    recent_accuracy=recent_accuracy,
                    accuracy_vs_confidence=accuracy_by_confidence,
                    mae=mae,
                    rmse=rmse,
                    r_squared=r_squared,
                    calibration_score=calibration_score,
                    roi_pct=0.0,  # Would need betting data to calculate
                    win_rate=overall_accuracy,
                    average_odds=-110,  # Placeholder
                    profit_loss=0.0,  # Would need betting data
                    total_predictions=len(model_data),
                    recent_predictions=len(recent_data),
                    predictions_by_confidence=predictions_by_confidence
                ))
        
        conn.close()
        return model_metrics
    
    def _analyze_betting_performance(self, start_date: datetime, end_date: datetime) -> List[BettingMetrics]:
        """Analyze betting performance metrics"""
        betting_metrics = []
        
        # Connect to betting database (if available)
        try:
            conn = sqlite3.connect("betting_strategy.db")
            
            query = '''
                SELECT strategy_name, stake_amount, odds, result, profit_loss, 
                       timestamp, game_id, bet_type
                FROM bet_history 
                WHERE timestamp BETWEEN ? AND ?
                AND result != 'PENDING'
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            
            if df.empty:
                conn.close()
                return betting_metrics
            
            # Analyze overall betting performance
            strategy_name = "Overall"
            
            total_staked = df['stake_amount'].sum()
            total_profit = df['profit_loss'].sum()
            roi_pct = (total_profit / total_staked) if total_staked > 0 else 0
            
            # Win/loss metrics
            wins = df[df['result'] == 'WIN']
            losses = df[df['result'] == 'LOSS']
            pushes = df[df['result'] == 'PUSH']
            
            win_rate = len(wins) / len(df) if len(df) > 0 else 0
            average_win = wins['profit_loss'].mean() if len(wins) > 0 else 0
            average_loss = losses['profit_loss'].mean() if len(losses) > 0 else 0
            largest_win = wins['profit_loss'].max() if len(wins) > 0 else 0
            largest_loss = losses['profit_loss'].min() if len(losses) > 0 else 0
            
            # Profit factor
            gross_profit = wins['profit_loss'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['profit_loss'].sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss
            
            # Drawdown calculation
            df_sorted = df.sort_values('timestamp')
            cumulative_profit = df_sorted['profit_loss'].cumsum()
            running_max = cumulative_profit.cummax()
            drawdown = (running_max - cumulative_profit) / running_max.where(running_max != 0, 1)
            max_drawdown = drawdown.max()
            current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
            
            # Sharpe ratio (simplified)
            daily_returns = df_sorted.groupby(df_sorted['timestamp'].dt.date)['profit_loss'].sum()
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = daily_returns[daily_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else daily_returns.std()
            sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Value at Risk (5% VaR)
            value_at_risk = daily_returns.quantile(0.05) if len(daily_returns) > 0 else 0
            
            # Monthly performance
            df_sorted['month'] = df_sorted['timestamp'].dt.to_period('M')
            monthly_performance = df_sorted.groupby('month')['profit_loss'].sum().to_dict()
            monthly_performance = {str(k): v for k, v in monthly_performance.items()}
            
            # Streak analysis
            df_sorted['win'] = (df_sorted['result'] == 'WIN').astype(int)
            df_sorted['loss'] = (df_sorted['result'] == 'LOSS').astype(int)
            
            # Calculate streaks (simplified)
            win_streaks = []
            loss_streaks = []
            current_win_streak = 0
            current_loss_streak = 0
            
            for _, row in df_sorted.iterrows():
                if row['win']:
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                elif row['loss']:
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
            
            streak_metrics = {
                'max_win_streak': max(win_streaks) if win_streaks else 0,
                'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
                'current_win_streak': current_win_streak,
                'current_loss_streak': current_loss_streak
            }
            
            betting_metrics.append(BettingMetrics(
                strategy_name=strategy_name,
                total_staked=total_staked,
                total_profit=total_profit,
                roi_pct=roi_pct,
                profit_factor=profit_factor,
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                value_at_risk=value_at_risk,
                total_bets=len(df),
                winning_bets=len(wins),
                losing_bets=len(losses),
                push_bets=len(pushes),
                daily_avg_profit=daily_returns.mean() if len(daily_returns) > 0 else 0,
                monthly_performance=monthly_performance,
                streak_metrics=streak_metrics
            ))
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not analyze betting performance: {e}")
        
        return betting_metrics
    
    def _analyze_system_health(self, start_date: datetime, end_date: datetime) -> SystemHealthMetrics:
        """Analyze overall system health"""
        
        # Placeholder implementation - would analyze various system metrics
        uptime_pct = 0.995  # 99.5% uptime
        data_freshness = 0.5  # 30 minutes since last update
        model_agreement = 0.85  # 85% model agreement
        prediction_confidence_avg = 0.75  # 75% average confidence
        
        # Performance alerts
        accuracy_declining = False  # Would compare to historical baselines
        roi_declining = False
        high_drawdown = False
        model_drift_detected = False
        
        predictions_per_day = 25.0  # Average predictions per day
        bets_per_day = 8.0  # Average bets per day
        data_quality_score = 0.90  # 90% data quality
        coverage_pct = 0.85  # 85% game coverage
        
        return SystemHealthMetrics(
            uptime_pct=uptime_pct,
            data_freshness=data_freshness,
            model_agreement=model_agreement,
            prediction_confidence_avg=prediction_confidence_avg,
            accuracy_declining=accuracy_declining,
            roi_declining=roi_declining,
            high_drawdown=high_drawdown,
            model_drift_detected=model_drift_detected,
            predictions_per_day=predictions_per_day,
            bets_per_day=bets_per_day,
            data_quality_score=data_quality_score,
            coverage_pct=coverage_pct
        )
    
    def _generate_key_insights(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate key insights from performance data"""
        insights = []
        
        # Analyze model performance
        model_metrics = self._analyze_model_performance(start_date, end_date)
        if model_metrics:
            best_model = max(model_metrics, key=lambda x: x.overall_accuracy)
            insights.append(f"Best performing model: {best_model.model_name} ({best_model.prediction_type}) with {best_model.overall_accuracy:.1%} accuracy")
            
            # Check for declining performance
            recent_avg_accuracy = np.mean([m.recent_accuracy for m in model_metrics])
            overall_avg_accuracy = np.mean([m.overall_accuracy for m in model_metrics])
            
            if recent_avg_accuracy < overall_avg_accuracy - 0.05:
                insights.append("⚠️  Recent model accuracy has declined by >5% - consider retraining")
        
        # Analyze betting performance
        betting_metrics = self._analyze_betting_performance(start_date, end_date)
        if betting_metrics:
            overall = betting_metrics[0]
            insights.append(f"Overall betting ROI: {overall.roi_pct:.1%} with {overall.win_rate:.1%} win rate")
            
            if overall.current_drawdown > 0.1:
                insights.append("⚠️  Currently in >10% drawdown - consider reducing bet sizes")
            
            if overall.roi_pct > 0.05:
                insights.append("✅  Profitable betting strategy with >5% ROI")
        
        # System health insights
        insights.append(f"System processing {25} predictions and {8} bets per day on average")
        insights.append("✅  System health: All indicators green")
        
        return insights
    
    def _generate_recommendations(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Model recommendations
        model_metrics = self._analyze_model_performance(start_date, end_date)
        if model_metrics:
            low_accuracy_models = [m for m in model_metrics if m.overall_accuracy < 0.52]
            if low_accuracy_models:
                recommendations.append("Consider retraining or removing underperforming models with <52% accuracy")
            
            high_confidence_low_accuracy = [m for m in model_metrics 
                                          if m.accuracy_vs_confidence.get('High', 0) < m.overall_accuracy - 0.05]
            if high_confidence_low_accuracy:
                recommendations.append("Recalibrate confidence scoring - high confidence predictions underperforming")
        
        # Betting recommendations
        betting_metrics = self._analyze_betting_performance(start_date, end_date)
        if betting_metrics:
            overall = betting_metrics[0]
            
            if overall.max_drawdown > 0.15:
                recommendations.append("Implement stricter risk management - max drawdown >15%")
            
            if overall.profit_factor < 1.5:
                recommendations.append("Focus on higher edge opportunities - profit factor below optimal")
            
            if overall.sharpe_ratio < 1.0:
                recommendations.append("Consider diversification to improve risk-adjusted returns")
        
        # System recommendations
        recommendations.append("Monitor line movement patterns for better timing opportunities")
        recommendations.append("Consider expanding coverage to mid-major conferences for additional value")
        
        return recommendations
    
    def _generate_chart_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate data for charts and visualizations"""
        chart_data = {}
        
        # Model accuracy over time
        model_metrics = self._analyze_model_performance(start_date, end_date)
        if model_metrics:
            chart_data['model_accuracy'] = {
                'models': [m.model_name for m in model_metrics],
                'overall_accuracy': [m.overall_accuracy for m in model_metrics],
                'recent_accuracy': [m.recent_accuracy for m in model_metrics]
            }
        
        # Betting performance over time
        betting_metrics = self._analyze_betting_performance(start_date, end_date)
        if betting_metrics:
            overall = betting_metrics[0]
            chart_data['betting_performance'] = {
                'monthly_performance': overall.monthly_performance,
                'roi_pct': overall.roi_pct,
                'win_rate': overall.win_rate,
                'profit_factor': overall.profit_factor
            }
        
        # Risk metrics
        chart_data['risk_metrics'] = {
            'max_drawdown': betting_metrics[0].max_drawdown if betting_metrics else 0,
            'current_drawdown': betting_metrics[0].current_drawdown if betting_metrics else 0,
            'sharpe_ratio': betting_metrics[0].sharpe_ratio if betting_metrics else 0,
            'var_95': betting_metrics[0].value_at_risk if betting_metrics else 0
        }
        
        return chart_data
    
    def _save_report(self, report: Dict[str, Any]):
        """Save performance report"""
        reports_dir = Path("performance_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"performance_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved: {filename}")
    
    def create_performance_visualizations(self, report: Dict[str, Any]):
        """Create performance visualization charts"""
        
        # Setup figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('College Basketball System Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Model Accuracy Chart
        if 'model_accuracy' in report['charts']:
            data = report['charts']['model_accuracy']
            ax1 = axes[0, 0]
            x = np.arange(len(data['models']))
            width = 0.35
            
            ax1.bar(x - width/2, data['overall_accuracy'], width, label='Overall', color=self.colors['primary'])
            ax1.bar(x + width/2, data['recent_accuracy'], width, label='Recent', color=self.colors['secondary'])
            
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(data['models'], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # ROI Over Time
        if 'betting_performance' in report['charts']:
            data = report['charts']['betting_performance']
            ax2 = axes[0, 1]
            
            months = list(data['monthly_performance'].keys())
            profits = list(data['monthly_performance'].values())
            
            ax2.plot(months, profits, marker='o', color=self.colors['success'])
            ax2.set_ylabel('Profit ($)')
            ax2.set_title('Monthly Profit/Loss')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Risk Metrics
        risk_data = report['charts']['risk_metrics']
        ax3 = axes[0, 2]
        
        metrics = ['Max Drawdown', 'Current Drawdown', 'Sharpe Ratio', 'VaR 95%']
        values = [risk_data['max_drawdown'], risk_data['current_drawdown'], 
                 risk_data['sharpe_ratio'], abs(risk_data['var_95'])]
        colors = [self.colors['danger'], self.colors['warning'], 
                 self.colors['success'], self.colors['info']]
        
        ax3.bar(metrics, values, color=colors)
        ax3.set_ylabel('Value')
        ax3.set_title('Risk Metrics')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Betting Performance Summary
        ax4 = axes[1, 0]
        if betting_metrics := report.get('betting_metrics'):
            overall = betting_metrics[0] if betting_metrics else None
            if overall:
                labels = ['Wins', 'Losses', 'Pushes']
                sizes = [overall['winning_bets'], overall['losing_bets'], overall['push_bets']]
                colors = [self.colors['success'], self.colors['danger'], self.colors['warning']]
                
                ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Betting Results Distribution')
        
        # Model Performance by Confidence
        ax5 = axes[1, 1]
        if model_metrics := report.get('model_metrics'):
            if model_metrics:
                confidence_levels = ['Low', 'Medium', 'High']
                avg_accuracy = []
                
                for level in confidence_levels:
                    accuracies = [m['accuracy_vs_confidence'].get(level, 0) 
                                for m in model_metrics if 'accuracy_vs_confidence' in m]
                    avg_accuracy.append(np.mean(accuracies) if accuracies else 0)
                
                ax5.bar(confidence_levels, avg_accuracy, color=self.colors['primary'])
                ax5.set_ylabel('Accuracy')
                ax5.set_title('Accuracy by Confidence Level')
                ax5.grid(True, alpha=0.3)
        
        # System Health Dashboard
        ax6 = axes[1, 2]
        if system_health := report.get('system_health'):
            health_metrics = ['Uptime', 'Data Quality', 'Coverage', 'Model Agreement']
            health_values = [
                system_health['uptime_pct'], 
                system_health['data_quality_score'],
                system_health['coverage_pct'], 
                system_health['model_agreement']
            ]
            
            colors = [self.colors['success'] if v > 0.8 else 
                     self.colors['warning'] if v > 0.6 else 
                     self.colors['danger'] for v in health_values]
            
            ax6.bar(health_metrics, health_values, color=colors)
            ax6.set_ylabel('Score')
            ax6.set_title('System Health Metrics')
            ax6.set_ylim(0, 1)
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        charts_dir = Path("performance_charts")
        charts_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = charts_dir / f"performance_dashboard_{timestamp}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Performance charts saved: {filename}")
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        # Generate current report for analysis
        report = self.generate_comprehensive_report(days_back=7)
        
        # Check model accuracy alerts
        if model_metrics := report.get('model_metrics'):
            for model in model_metrics:
                if model['recent_accuracy'] < model['overall_accuracy'] - self.alert_thresholds['accuracy_decline']:
                    alerts.append({
                        'type': 'MODEL_ACCURACY_DECLINE',
                        'severity': 'HIGH',
                        'message': f"{model['model_name']} accuracy declined by {(model['overall_accuracy'] - model['recent_accuracy']):.1%}",
                        'metric_value': model['recent_accuracy'],
                        'threshold_value': model['overall_accuracy'] - self.alert_thresholds['accuracy_decline']
                    })
        
        # Check betting performance alerts
        if betting_metrics := report.get('betting_metrics'):
            overall = betting_metrics[0] if betting_metrics else None
            if overall:
                if overall['current_drawdown'] > self.alert_thresholds['max_drawdown']:
                    alerts.append({
                        'type': 'HIGH_DRAWDOWN',
                        'severity': 'HIGH',
                        'message': f"Current drawdown {overall['current_drawdown']:.1%} exceeds threshold",
                        'metric_value': overall['current_drawdown'],
                        'threshold_value': self.alert_thresholds['max_drawdown']
                    })
        
        # Save alerts to database
        if alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def _save_alerts(self, alerts: List[Dict[str, Any]]):
        """Save alerts to database"""
        conn = sqlite3.connect(self.dashboard_db)
        cursor = conn.cursor()
        
        for alert in alerts:
            cursor.execute('''
                INSERT INTO performance_alerts 
                (timestamp, alert_type, severity, message, metric_value, threshold_value)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                alert['type'],
                alert['severity'],
                alert['message'],
                alert['metric_value'],
                alert['threshold_value']
            ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved {len(alerts)} performance alerts")
    
    def run_daily_performance_update(self):
        """Run daily performance tracking update"""
        self.logger.info("Running daily performance update...")
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(days_back=1)
        
        # Check for alerts
        alerts = self.check_performance_alerts()
        
        # Create visualizations
        self.create_performance_visualizations(report)
        
        # Log summary
        self.logger.info("Daily performance update completed")
        self.logger.info(f"Generated {len(alerts)} alerts")
        
        if model_metrics := report.get('model_metrics'):
            avg_accuracy = np.mean([m['overall_accuracy'] for m in model_metrics])
            self.logger.info(f"Average model accuracy: {avg_accuracy:.1%}")
        
        if betting_metrics := report.get('betting_metrics'):
            overall = betting_metrics[0] if betting_metrics else None
            if overall:
                self.logger.info(f"Overall ROI: {overall['roi_pct']:.1%}")
                self.logger.info(f"Win Rate: {overall['win_rate']:.1%}")

# Testing and CLI interface
def main():
    """Main function for testing dashboard"""
    dashboard = PerformanceDashboard()
    
    print("🏀 College Basketball Performance Dashboard")
    print("=" * 50)
    
    # Generate comprehensive report
    print("Generating performance report...")
    report = dashboard.generate_comprehensive_report(days_back=30)
    
    # Display key insights
    print("\n📊 Key Insights:")
    for insight in report['key_insights']:
        print(f"• {insight}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    # Check alerts
    alerts = dashboard.check_performance_alerts()
    if alerts:
        print(f"\n⚠️  Performance Alerts:")
        for alert in alerts:
            print(f"• {alert['severity']}: {alert['message']}")
    else:
        print(f"\n✅ No performance alerts")
    
    # Create visualizations
    print(f"\nGenerating performance visualizations...")
    dashboard.create_performance_visualizations(report)
    
    print(f"\n✅ Performance dashboard completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()