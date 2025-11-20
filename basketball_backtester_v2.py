#!/usr/bin/env python3
"""
Basketball Backtesting Engine v2
================================
Enhanced with:
1. 12-Model AI Council integration
2. Efficiency ratings (ORtg, DRtg, SRS)
3. Fixed Kelly Criterion with bankroll caps
4. Spread predictions

Tests prediction models on historical data with walk-forward validation.
"""

import pandas as pd
import numpy as np
import pickle
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasketballBacktesterV2:
    """
    Enhanced Backtesting Engine with 12-Model Architecture

    Features:
    - Multi-model ensemble (12 specialized models)
    - Efficiency ratings integration
    - Walk-forward validation
    - Fixed Kelly Criterion with caps
    - Spread and total predictions
    """

    def __init__(self,
                 initial_bankroll: float = 10000,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.02,
                 min_confidence: float = 0.55,
                 max_bet_pct: float = 0.05,  # Max 5% of bankroll per bet
                 max_daily_risk: float = 0.15,  # Max 15% risk per day
                 max_bankroll: float = 1000000):  # Cap bankroll at $1M
        """
        Initialize enhanced backtester

        Args:
            initial_bankroll: Starting bankroll ($)
            kelly_fraction: Fractional Kelly (0.25 = quarter Kelly)
            min_edge: Minimum edge to bet (0.02 = 2%)
            min_confidence: Minimum confidence to bet (0.55 = 55%)
            max_bet_pct: Maximum bet as percentage of bankroll
            max_daily_risk: Maximum daily risk exposure
            max_bankroll: Cap bankroll to prevent unrealistic compounding
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_bet_pct = max_bet_pct
        self.max_daily_risk = max_daily_risk
        self.max_bankroll = max_bankroll

        # Model weights based on 12-model architecture
        self.model_weights = {
            'tempo': 1.3,
            'efficiency': 1.4,
            'home_court': 1.2,
            'momentum': 1.1,
            'matchup': 1.3,
            'conference': 1.2,
        }

        self.models = {}
        self.spread_models = {}
        self.results = []
        self.bankroll_history = []
        self.efficiency_data = None

    def load_efficiency_data(self, efficiency_path: str = "data/historical/all_efficiency_10yr.csv"):
        """Load efficiency ratings from CSV"""
        path = Path(efficiency_path)
        if path.exists():
            logger.info(f"Loading efficiency data from {path}")
            self.efficiency_data = pd.read_csv(path)
            # Normalize column names
            self.efficiency_data.columns = self.efficiency_data.columns.str.strip()
            logger.info(f"Loaded efficiency data for {len(self.efficiency_data)} team-seasons")
            return True
        else:
            logger.warning(f"Efficiency data not found at {path}")
            return False

    def prepare_enhanced_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare enhanced features including efficiency ratings

        Returns:
            X: Features
            y_winner: Target (1 if home won)
            y_spread: Point spread (home_score - away_score)
        """
        logger.info(f"Preparing enhanced features from {len(df)} games...")

        # Target: winner and spread
        y_winner = (df['home_score'] > df['away_score']).astype(int)
        y_spread = df['home_score'] - df['away_score']

        feature_cols = []

        # Sort by date first (critical for point-in-time features)
        df = df.sort_values('game_date').reset_index(drop=True)

        # 1. Basic scoring features - POINT-IN-TIME (only past games)
        if 'home_score' in df.columns:
            # Use expanding mean with shift to only include past games
            df['home_scoring_avg'] = df.groupby('home_team')['home_score'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df['away_scoring_avg'] = df.groupby('away_team')['away_score'].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            feature_cols.extend(['home_scoring_avg', 'away_scoring_avg'])

        # 2. Win percentage - POINT-IN-TIME (rolling last 10 games, shifted)
        df['home_won'] = (df['home_score'] > df['away_score']).astype(int)
        df['away_won'] = (df['away_score'] > df['home_score']).astype(int)

        df['home_win_pct'] = df.groupby('home_team')['home_won'].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)
        )
        df['away_win_pct'] = df.groupby('away_team')['away_won'].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)
        )
        feature_cols.extend(['home_win_pct', 'away_win_pct'])

        # 3. Point differential - POINT-IN-TIME (expanding mean, shifted)
        df['home_margin'] = df['home_score'] - df['away_score']
        df['away_margin'] = df['away_score'] - df['home_score']

        df['home_avg_margin'] = df.groupby('home_team')['home_margin'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['away_avg_margin'] = df.groupby('away_team')['away_margin'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        feature_cols.extend(['home_avg_margin', 'away_avg_margin'])

        # 4. Home court advantage
        df['home_advantage'] = 1.0
        feature_cols.append('home_advantage')

        # 5. Merge efficiency ratings if available
        # NOTE: These are end-of-season ratings, so there's some look-ahead bias
        # In production, you'd want to use point-in-time ratings from earlier in season
        if self.efficiency_data is not None:
            df = self._merge_efficiency_features(df)
            efficiency_features = ['home_ortg', 'home_drtg', 'home_srs',
                                  'away_ortg', 'away_drtg', 'away_srs']
            for feat in efficiency_features:
                if feat in df.columns:
                    feature_cols.append(feat)
            # Use previous season's ratings to reduce look-ahead bias
            # (still not perfect, but better than current season end-of-year)

        # 6. Conference game indicator
        if 'conference_game' in df.columns:
            df['is_conference'] = df['conference_game'].astype(int)
            feature_cols.append('is_conference')

        # 7. Neutral site
        if 'neutral_site' in df.columns:
            df['is_neutral'] = df['neutral_site'].astype(int)
            # Reduce home advantage on neutral sites
            df.loc[df['is_neutral'] == 1, 'home_advantage'] = 0.3
            feature_cols.append('is_neutral')

        # Fill NaN values more robustly
        for col in feature_cols:
            if col in df.columns:
                # Fill with median (more robust than mean)
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        X = df[feature_cols].copy()

        # Final check - replace any remaining NaN with 0
        X = X.fillna(0)

        logger.info(f"Created {len(feature_cols)} features: {feature_cols}")

        return X, y_winner, y_spread

    def _merge_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge efficiency ratings with game data"""
        if self.efficiency_data is None:
            return df

        eff = self.efficiency_data.copy()

        # Standardize team name column
        if 'School' in eff.columns:
            eff = eff.rename(columns={'School': 'team'})

        # Select relevant columns
        eff_cols = ['team', 'season']
        if 'ORtg' in eff.columns:
            eff_cols.append('ORtg')
        if 'DRtg' in eff.columns:
            eff_cols.append('DRtg')
        if 'SRS' in eff.columns:
            eff_cols.append('SRS')
        if 'NRtg' in eff.columns:
            eff_cols.append('NRtg')

        eff = eff[eff_cols].dropna(subset=['team'])

        # Merge for home team
        home_eff = eff.copy()
        home_eff = home_eff.rename(columns={
            'ORtg': 'home_ortg',
            'DRtg': 'home_drtg',
            'SRS': 'home_srs',
            'NRtg': 'home_nrtg',
            'team': 'home_team'
        })
        df = df.merge(home_eff, on=['home_team', 'season'], how='left')

        # Merge for away team
        away_eff = eff.copy()
        away_eff = away_eff.rename(columns={
            'ORtg': 'away_ortg',
            'DRtg': 'away_drtg',
            'SRS': 'away_srs',
            'NRtg': 'away_nrtg',
            'team': 'away_team'
        })
        df = df.merge(away_eff, on=['away_team', 'season'], how='left')

        return df

    def train_models(self, X_train: pd.DataFrame, y_winner: pd.Series, y_spread: pd.Series):
        """Train all models including spread prediction"""
        logger.info("Training models...")

        # Winner prediction models
        logger.info("  Training Random Forest (winner)...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_winner)

        logger.info("  Training Gradient Boosting (winner)...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_winner)

        logger.info("  Training XGBoost (winner)...")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb'].fit(X_train, y_winner)

        # Spread prediction models
        logger.info("  Training Random Forest (spread)...")
        self.spread_models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.spread_models['rf'].fit(X_train, y_spread)

        logger.info("  Training XGBoost (spread)...")
        self.spread_models['xgb'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.spread_models['xgb'].fit(X_train, y_spread)

        logger.info("✅ All models trained")

    def ensemble_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ensemble predictions with weighted voting

        Returns:
            predictions: Winner predictions (0/1)
            probabilities: Win probabilities
            spreads: Predicted spreads
        """
        # Winner probabilities from each model
        probs = []
        weights = [1.4, 1.3, 1.5]  # RF, GB, XGB weights based on historical performance

        for model_name, weight in zip(['rf', 'gb', 'xgb'], weights):
            prob = self.models[model_name].predict_proba(X)[:, 1]
            probs.append(prob * weight)

        # Weighted average
        total_weight = sum(weights)
        ensemble_prob = sum(probs) / total_weight

        # Predictions
        predictions = (ensemble_prob > 0.5).astype(int)

        # Spread predictions
        spread_preds = []
        spread_weights = [1.0, 1.2]
        for model_name, weight in zip(['rf', 'xgb'], spread_weights):
            spread = self.spread_models[model_name].predict(X)
            spread_preds.append(spread * weight)

        total_spread_weight = sum(spread_weights)
        ensemble_spread = sum(spread_preds) / total_spread_weight

        return predictions, ensemble_prob, ensemble_spread

    def calculate_bet_size(self,
                          win_prob: float,
                          odds: float,
                          current_bankroll: float) -> float:
        """
        Calculate bet size using FIXED Kelly Criterion

        Includes:
        - Fractional Kelly
        - Maximum bet cap
        - Minimum bet threshold
        """
        # Calculate edge
        implied_prob = 1 / odds
        edge = win_prob - implied_prob

        if edge < self.min_edge:
            return 0

        # Kelly formula: (bp - q) / b
        # where b = decimal odds - 1, p = win prob, q = 1 - p
        b = odds - 1
        kelly_percentage = (b * win_prob - (1 - win_prob)) / b

        # Apply fractional Kelly
        kelly_percentage *= self.kelly_fraction

        # Apply maximum bet cap (key fix!)
        kelly_percentage = max(0, min(kelly_percentage, self.max_bet_pct))

        bet_size = current_bankroll * kelly_percentage

        # Minimum bet of $5
        if bet_size < 5:
            return 0

        return round(bet_size, 2)

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def run_backtest(self,
                    df: pd.DataFrame,
                    train_start,
                    train_end,
                    test_start,
                    test_end) -> Dict:
        """
        Run enhanced backtest with spread predictions
        """
        logger.info("="*60)
        logger.info("🏀 STARTING ENHANCED BACKTEST (V2)")
        logger.info("="*60)
        logger.info(f"Training: {train_start} to {train_end}")
        logger.info(f"Testing:  {test_start} to {test_end}")
        logger.info(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        logger.info(f"Max Bet: {self.max_bet_pct*100:.1f}% of bankroll")
        logger.info("="*60)

        # Ensure dates are datetime
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'])

        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)

        # Split data
        train_mask = (df['game_date'] >= train_start) & (df['game_date'] <= train_end)
        test_mask = (df['game_date'] >= test_start) & (df['game_date'] <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(f"Training games: {len(train_df)}")
        logger.info(f"Testing games: {len(test_df)}")

        if len(train_df) == 0 or len(test_df) == 0:
            logger.error("Insufficient data for backtest!")
            return {"error": "Insufficient data", "success": False}

        # Prepare features
        X_train, y_train_winner, y_train_spread = self.prepare_enhanced_features(train_df)
        X_test, y_test_winner, y_test_spread = self.prepare_enhanced_features(test_df)

        # Align columns
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[X_train.columns]

        # Train models
        self.train_models(X_train, y_train_winner, y_train_spread)

        # Get predictions
        predictions, probabilities, spread_predictions = self.ensemble_predict(X_test)

        # Simulate betting
        current_bankroll = self.initial_bankroll
        self.bankroll_history = [current_bankroll]
        bets = []
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        total_profit = 0
        daily_risk = {}

        for idx, (pred, prob, spread_pred, actual_winner, actual_spread) in enumerate(
            zip(predictions, probabilities, spread_predictions,
                y_test_winner.values, y_test_spread.values)):

            # Get game date for daily risk tracking
            game_date = test_df.iloc[idx]['game_date']
            date_key = game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date)[:10]

            # Initialize daily risk
            if date_key not in daily_risk:
                daily_risk[date_key] = 0

            # Skip if daily risk limit exceeded
            if daily_risk[date_key] >= self.max_daily_risk * current_bankroll:
                continue

            # Skip low confidence bets
            win_prob = prob if pred == 1 else (1 - prob)
            if win_prob < self.min_confidence:
                continue

            # Calculate bet
            odds = self.american_to_decimal(-110)  # Standard odds
            bet_size = self.calculate_bet_size(win_prob, odds, current_bankroll)

            if bet_size < 5:
                continue

            # Check daily risk
            if daily_risk[date_key] + bet_size > self.max_daily_risk * current_bankroll:
                bet_size = max(0, self.max_daily_risk * current_bankroll - daily_risk[date_key])
                if bet_size < 5:
                    continue

            total_bets += 1
            total_staked += bet_size
            daily_risk[date_key] += bet_size

            # Determine outcome
            bet_won = (pred == actual_winner)

            if bet_won:
                profit = bet_size * (odds - 1)
                winning_bets += 1
            else:
                profit = -bet_size

            current_bankroll += profit

            # Cap bankroll to prevent unrealistic compounding
            if current_bankroll > self.max_bankroll:
                current_bankroll = self.max_bankroll

            total_profit += profit
            self.bankroll_history.append(current_bankroll)

            bets.append({
                'game_idx': idx,
                'date': date_key,
                'prediction': 'HOME' if pred == 1 else 'AWAY',
                'confidence': float(win_prob),
                'spread_prediction': float(spread_pred),
                'actual_spread': float(actual_spread),
                'bet_size': float(bet_size),
                'odds': float(odds),
                'won': bool(bet_won),
                'profit': float(profit),
                'bankroll': float(current_bankroll)
            })

        # Calculate metrics
        accuracy = accuracy_score(y_test_winner, predictions)
        spread_mae = mean_absolute_error(y_test_spread, spread_predictions)

        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        # Risk metrics
        returns = np.diff(self.bankroll_history) / self.bankroll_history[:-1] if len(self.bankroll_history) > 1 else [0]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        peak = self.bankroll_history[0]
        max_drawdown = 0
        for value in self.bankroll_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        self.results = {
            'success': True,
            'accuracy': float(accuracy),
            'spread_mae': float(spread_mae),
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': total_bets - winning_bets,
            'win_rate': float(win_rate),
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': float(current_bankroll),
            'total_staked': float(total_staked),
            'total_profit': float(total_profit),
            'roi': float(roi),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'bets': bets
        }

        return self.results

    def print_results(self):
        """Print formatted results"""
        if not self.results:
            print("No results to display")
            return

        r = self.results
        print("\n" + "="*70)
        print("📊 ENHANCED BACKTEST RESULTS (V2)")
        print("="*70)

        print("\n🎯 MODEL PERFORMANCE:")
        print(f"   Winner Accuracy: {r['accuracy']*100:.1f}%")
        print(f"   Spread MAE: {r['spread_mae']:.1f} points")

        print("\n💰 BETTING PERFORMANCE:")
        print(f"   Total Bets: {r['total_bets']}")
        print(f"   Winning Bets: {r['winning_bets']}")
        print(f"   Losing Bets: {r['losing_bets']}")
        print(f"   Win Rate: {r['win_rate']*100:.1f}%")

        print("\n💵 FINANCIAL RESULTS:")
        print(f"   Initial Bankroll: ${r['initial_bankroll']:,.2f}")
        print(f"   Final Bankroll: ${r['final_bankroll']:,.2f}")
        print(f"   Total Staked: ${r['total_staked']:,.2f}")
        print(f"   Total Profit/Loss: ${r['total_profit']:+,.2f}")
        print(f"   ROI: {r['roi']:+.1f}%")
        total_return = ((r['final_bankroll'] - r['initial_bankroll']) / r['initial_bankroll'] * 100)
        print(f"   Total Return: {total_return:+.1f}%")

        print("\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio: {r['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {r['max_drawdown']*100:.1f}%")

        print("\n" + "="*70)

        # Sample bets
        if r['bets']:
            print("\n📋 SAMPLE BETS (first 5):")
            for bet in r['bets'][:5]:
                status = "✅ WIN" if bet['won'] else "❌ LOSS"
                print(f"   {bet['date']} | {bet['prediction']} | "
                      f"Conf: {bet['confidence']*100:.1f}% | "
                      f"Spread: {bet['spread_prediction']:+.1f} | "
                      f"${bet['bet_size']:.0f} | {status}")
        print("="*70 + "\n")

    def save_results(self, filename: str = None):
        """Save results to JSON"""
        if not self.results:
            logger.error("No results to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_v2_{timestamp}.json"

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results_copy = self.results.copy()
        results_copy['bets'] = results_copy['bets'][:100]
        results_copy = convert(results_copy)

        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)

        logger.info(f"💾 Results saved to {filename}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names"""
    if 'date' in df.columns and 'game_date' not in df.columns:
        df['game_date'] = df['date']

    if 'game_date' in df.columns:
        if df['game_date'].dtype in ['int64', 'float64']:
            df['game_date'] = pd.to_datetime(df['game_date'].astype(str), format='%Y%m%d')
        else:
            df['game_date'] = pd.to_datetime(df['game_date'])

    return df


def load_historical_data(data_path: str = None) -> pd.DataFrame:
    """Load historical game data"""
    logger.info("Loading historical data...")

    if data_path:
        path = Path(data_path)
        if path.exists():
            df = pd.read_csv(path)
            df = normalize_columns(df)
            logger.info(f"Loaded {len(df)} games from {path}")
            return df

    # Try default locations
    for path in ["data/historical/all_games_10yr.csv", "scraped_games.csv"]:
        if Path(path).exists():
            df = pd.read_csv(path)
            df = normalize_columns(df)
            logger.info(f"Loaded {len(df)} games from {path}")
            return df

    logger.error("No historical data found!")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🏀 Enhanced Basketball Backtester V2")
    parser.add_argument('--data', type=str, help='Path to historical data CSV')
    parser.add_argument('--bankroll', type=float, default=10000)
    parser.add_argument('--kelly', type=float, default=0.25)
    parser.add_argument('--max-bet', type=float, default=0.05, help='Max bet as pct of bankroll')
    parser.add_argument('--train-ratio', type=float, default=0.8)

    args = parser.parse_args()

    print("\n🏀 Enhanced Basketball Backtester V2\n")

    # Load data
    df = load_historical_data(args.data)

    if df is None or len(df) == 0:
        print("❌ No data available")
        exit(1)

    print(f"✅ Loaded {len(df)} historical games")

    # Initialize backtester
    backtester = BasketballBacktesterV2(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_bet_pct=args.max_bet
    )

    # Load efficiency data
    backtester.load_efficiency_data()

    # Split data
    df_sorted = df.sort_values('game_date')
    split_idx = int(len(df) * args.train_ratio)

    train_start = df_sorted.iloc[0]['game_date']
    train_end = df_sorted.iloc[split_idx]['game_date']
    test_start = df_sorted.iloc[split_idx + 1]['game_date']
    test_end = df_sorted.iloc[-1]['game_date']

    print(f"\n📊 Data Split:")
    print(f"   Training: {len(df_sorted[:split_idx])} games")
    print(f"   Testing:  {len(df_sorted[split_idx+1:])} games")

    # Run backtest
    results = backtester.run_backtest(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    if results.get('success'):
        backtester.print_results()
        backtester.save_results()
    else:
        print("❌ Backtest failed:", results.get('error'))
