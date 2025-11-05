#!/usr/bin/env python3
"""
Advanced Machine Learning Pipeline for College Basketball
========================================================

Enhanced ML pipeline featuring:
- Ensemble methods (XGBoost, Random Forest, Neural Networks)
- Advanced feature engineering
- Time series analysis for momentum
- Player-level impact modeling
- Automated hyperparameter tuning
- Cross-validation with temporal splits
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import joblib
from pathlib import Path

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - deep learning models disabled")

@dataclass
class FeatureSet:
    """Comprehensive feature set for ML models"""
    # Team strength features
    team_efficiency_features: Dict[str, float]
    opponent_efficiency_features: Dict[str, float]
    
    # Momentum and recent form
    momentum_features: Dict[str, float]
    recent_performance_features: Dict[str, float]
    
    # Advanced metrics
    advanced_stats_features: Dict[str, float]
    matchup_features: Dict[str, float]
    
    # Contextual features
    situational_features: Dict[str, float]
    environmental_features: Dict[str, float]
    
    # Player-level aggregated features
    player_impact_features: Dict[str, float]
    
    # Meta features
    confidence_features: Dict[str, float]

@dataclass
class ModelPrediction:
    """Individual model prediction with confidence"""
    model_name: str
    win_probability: float
    point_spread: float
    total_points: float
    confidence: float
    feature_importance: Dict[str, float]

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction"""
    final_win_probability: float
    final_point_spread: float
    final_total_points: float
    model_agreement: float
    prediction_confidence: float
    individual_predictions: List[ModelPrediction]

class AdvancedFeatureEngineering:
    """Advanced feature engineering for basketball predictions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        
    def create_comprehensive_features(self, team_data: Dict, opponent_data: Dict, 
                                    game_context: Dict) -> FeatureSet:
        """Create comprehensive feature set"""
        
        # Team efficiency features
        team_eff = self._create_efficiency_features(team_data)
        opp_eff = self._create_efficiency_features(opponent_data, prefix="opp_")
        
        # Momentum features
        momentum = self._create_momentum_features(team_data, opponent_data)
        
        # Recent performance (last 10 games)
        recent = self._create_recent_performance_features(team_data, opponent_data)
        
        # Advanced statistics
        advanced = self._create_advanced_stats_features(team_data, opponent_data)
        
        # Matchup-specific features
        matchup = self._create_matchup_features(team_data, opponent_data)
        
        # Situational context
        situational = self._create_situational_features(game_context)
        
        # Environmental factors
        environmental = self._create_environmental_features(game_context)
        
        # Player impact aggregation
        player_impact = self._create_player_impact_features(team_data, opponent_data)
        
        # Confidence and meta features
        confidence = self._create_confidence_features(team_data, opponent_data, game_context)
        
        return FeatureSet(
            team_efficiency_features=team_eff,
            opponent_efficiency_features=opp_eff,
            momentum_features=momentum,
            recent_performance_features=recent,
            advanced_stats_features=advanced,
            matchup_features=matchup,
            situational_features=situational,
            environmental_features=environmental,
            player_impact_features=player_impact,
            confidence_features=confidence
        )
    
    def _create_efficiency_features(self, team_data: Dict, prefix: str = "") -> Dict[str, float]:
        """Create efficiency-based features"""
        return {
            f"{prefix}offensive_efficiency": team_data.get('offensive_efficiency', 100.0),
            f"{prefix}defensive_efficiency": team_data.get('defensive_efficiency', 100.0),
            f"{prefix}net_efficiency": team_data.get('net_efficiency', 0.0),
            f"{prefix}effective_fg_pct": team_data.get('effective_fg_pct', 0.5),
            f"{prefix}turnover_rate": team_data.get('turnover_rate', 0.15),
            f"{prefix}offensive_rebounding_pct": team_data.get('offensive_rebounding_pct', 0.3),
            f"{prefix}free_throw_rate": team_data.get('free_throw_rate', 0.25),
            f"{prefix}tempo": team_data.get('tempo', 68.0),
            f"{prefix}three_point_rate": team_data.get('three_point_rate', 0.35),
            f"{prefix}three_point_pct": team_data.get('three_point_pct', 0.33),
        }
    
    def _create_momentum_features(self, team_data: Dict, opponent_data: Dict) -> Dict[str, float]:
        """Create momentum and trend features"""
        return {
            'team_momentum_score': self._calculate_momentum(team_data),
            'opponent_momentum_score': self._calculate_momentum(opponent_data),
            'momentum_differential': self._calculate_momentum(team_data) - self._calculate_momentum(opponent_data),
            'team_ats_streak': team_data.get('ats_streak', 0),
            'team_over_under_streak': team_data.get('ou_streak', 0),
            'recent_close_games': team_data.get('close_games_last_10', 0),
            'team_road_momentum': team_data.get('road_momentum', 0),
            'team_home_momentum': team_data.get('home_momentum', 0),
        }
    
    def _calculate_momentum(self, team_data: Dict) -> float:
        """Calculate team momentum score"""
        recent_games = team_data.get('recent_results', [])
        if not recent_games:
            return 0.0
        
        # Weight recent games more heavily
        weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        momentum = 0
        total_weight = 0
        
        for i, result in enumerate(recent_games[:10]):
            weight = weights[i] if i < len(weights) else 0.1
            # Result should be a dict with win/loss and margin
            if result.get('win'):
                momentum += weight * (1.0 + result.get('margin', 0) / 20.0)
            else:
                momentum -= weight * (1.0 + abs(result.get('margin', 0)) / 20.0)
            total_weight += weight
        
        return momentum / total_weight if total_weight > 0 else 0
    
    def _create_recent_performance_features(self, team_data: Dict, opponent_data: Dict) -> Dict[str, float]:
        """Create recent performance features (last 10 games)"""
        return {
            'team_recent_offensive_eff': team_data.get('recent_offensive_eff', 100.0),
            'team_recent_defensive_eff': team_data.get('recent_defensive_eff', 100.0),
            'opp_recent_offensive_eff': opponent_data.get('recent_offensive_eff', 100.0),
            'opp_recent_defensive_eff': opponent_data.get('recent_defensive_eff', 100.0),
            'team_recent_three_pt_pct': team_data.get('recent_three_pt_pct', 0.33),
            'team_recent_ft_pct': team_data.get('recent_ft_pct', 0.75),
            'team_recent_turnover_rate': team_data.get('recent_turnover_rate', 0.15),
            'team_games_last_7_days': team_data.get('games_last_7_days', 0),
            'team_travel_miles': team_data.get('travel_miles', 0),
            'team_rest_days': team_data.get('rest_days', 1),
        }
    
    def _create_advanced_stats_features(self, team_data: Dict, opponent_data: Dict) -> Dict[str, float]:
        """Create advanced statistical features"""
        return {
            'pace_differential': team_data.get('tempo', 68) - opponent_data.get('tempo', 68),
            'experience_differential': team_data.get('experience_factor', 0.5) - opponent_data.get('experience_factor', 0.5),
            'bench_depth_differential': team_data.get('bench_strength', 0.3) - opponent_data.get('bench_strength', 0.3),
            'coaching_experience_diff': team_data.get('coach_experience', 5) - opponent_data.get('coach_experience', 5),
            'tournament_experience_diff': team_data.get('tournament_games', 0) - opponent_data.get('tournament_games', 0),
            'kenpom_differential': team_data.get('kenpom_rating', 0) - opponent_data.get('kenpom_rating', 0),
            'strength_of_schedule_diff': team_data.get('sos_rating', 50) - opponent_data.get('sos_rating', 50),
            'injury_impact_differential': team_data.get('injury_impact', 0) - opponent_data.get('injury_impact', 0),
        }
    
    def _create_matchup_features(self, team_data: Dict, opponent_data: Dict) -> Dict[str, float]:
        """Create matchup-specific features"""
        return {
            'size_matchup': team_data.get('avg_height', 78) - opponent_data.get('avg_height', 78),
            'speed_vs_pace': team_data.get('speed_rating', 50) - opponent_data.get('tempo', 68),
            'three_point_attack_vs_defense': team_data.get('three_pt_rate', 0.35) - opponent_data.get('three_pt_def_pct', 0.33),
            'rebounding_matchup': team_data.get('rebounding_rate', 50) - opponent_data.get('opp_rebounding_rate', 50),
            'turnover_creation_vs_protection': team_data.get('steal_rate', 0.1) - opponent_data.get('turnover_rate', 0.15),
            'free_throw_creation_vs_fouling': team_data.get('foul_drawing_rate', 0.2) - opponent_data.get('foul_rate', 0.2),
            'home_court_advantage': team_data.get('home_court_strength', 0.65) if team_data.get('is_home') else 0,
            'conference_familiarity': 1.0 if team_data.get('conference') == opponent_data.get('conference') else 0.0,
        }
    
    def _create_situational_features(self, game_context: Dict) -> Dict[str, float]:
        """Create situational context features"""
        return {
            'is_tournament': 1.0 if game_context.get('tournament_context') != 'regular_season' else 0.0,
            'is_march_madness': 1.0 if game_context.get('tournament_context') == 'march_madness' else 0.0,
            'is_conference_tournament': 1.0 if game_context.get('tournament_context') == 'conference_tournament' else 0.0,
            'is_neutral_site': 1.0 if game_context.get('neutral_site', False) else 0.0,
            'is_rivalry': 1.0 if game_context.get('is_rivalry', False) else 0.0,
            'game_importance': game_context.get('game_importance', 5.0),  # 1-10 scale
            'seed_differential': abs(game_context.get('home_seed', 8) - game_context.get('away_seed', 8)),
            'ranking_differential': abs(game_context.get('home_ranking', 25) - game_context.get('away_ranking', 25)),
        }
    
    def _create_environmental_features(self, game_context: Dict) -> Dict[str, float]:
        """Create environmental context features"""
        return {
            'venue_capacity': game_context.get('venue_capacity', 15000) / 20000,  # Normalize
            'time_of_day': game_context.get('game_hour', 19),  # Hour of day
            'is_weekend': 1.0 if game_context.get('is_weekend', False) else 0.0,
            'is_prime_time': 1.0 if game_context.get('is_prime_time', False) else 0.0,
            'tv_coverage_tier': game_context.get('tv_tier', 2),  # 1=national, 2=regional, 3=local
            'weather_impact': game_context.get('weather_impact', 0.0),  # For travel
            'altitude': game_context.get('altitude', 1000) / 5000,  # Normalized altitude
        }
    
    def _create_player_impact_features(self, team_data: Dict, opponent_data: Dict) -> Dict[str, float]:
        """Create player-level aggregated features"""
        return {
            'star_player_impact': team_data.get('star_player_rating', 5.0),
            'depth_score': team_data.get('rotation_depth', 8),
            'player_experience_avg': team_data.get('avg_player_experience', 2.0),
            'injury_adjusted_strength': team_data.get('healthy_strength_rating', 7.0),
            'fatigue_factor': team_data.get('fatigue_rating', 1.0),
            'minutes_distribution': team_data.get('minutes_concentration', 0.6),  # How concentrated minutes are
            'freshman_impact': team_data.get('freshman_minutes_pct', 0.2),
            'transfer_impact': team_data.get('transfer_impact', 0.0),
        }
    
    def _create_confidence_features(self, team_data: Dict, opponent_data: Dict, 
                                  game_context: Dict) -> Dict[str, float]:
        """Create confidence and meta features"""
        return {
            'data_completeness': min(len(team_data) / 50, 1.0),  # How complete our data is
            'historical_sample_size': min(team_data.get('games_played', 0) / 30, 1.0),
            'recency_of_data': max(0, 1 - game_context.get('days_since_last_update', 0) / 7),
            'model_certainty': team_data.get('prediction_confidence', 0.7),
            'line_movement_indicator': abs(game_context.get('line_movement', 0)) / 3.0,
            'betting_volume_indicator': game_context.get('betting_volume', 0.5),
            'sharp_money_indicator': 1.0 if game_context.get('sharp_money', False) else 0.0,
        }
    
    def features_to_vector(self, feature_set: FeatureSet) -> np.ndarray:
        """Convert feature set to numpy array"""
        all_features = []
        
        # Flatten all feature dictionaries
        for feature_dict in [
            feature_set.team_efficiency_features,
            feature_set.opponent_efficiency_features,
            feature_set.momentum_features,
            feature_set.recent_performance_features,
            feature_set.advanced_stats_features,
            feature_set.matchup_features,
            feature_set.situational_features,
            feature_set.environmental_features,
            feature_set.player_impact_features,
            feature_set.confidence_features
        ]:
            all_features.extend(list(feature_dict.values()))
        
        return np.array(all_features).reshape(1, -1)

class AdvancedMLPipeline:
    """Advanced machine learning pipeline with ensemble methods"""
    
    def __init__(self, db_path: str = "advanced_ml_models.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.feature_engineer = AdvancedFeatureEngineering()
        
        # Initialize models
        self.models = {}
        self.ensemble_weights = {}
        self._initialize_models()
        
        # Initialize database
        self._init_database()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        
        # XGBoost models
        self.models['xgb_spread'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.models['xgb_total'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Random Forest models
        self.models['rf_spread'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.models['rf_total'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Neural Network models
        self.models['mlp_spread'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        self.models['mlp_total'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        # Deep learning models (if available)
        if TENSORFLOW_AVAILABLE:
            self._initialize_deep_models()
        
        # Initial ensemble weights (will be optimized)
        self.ensemble_weights = {
            'xgb': 0.4,
            'rf': 0.3,
            'mlp': 0.2,
            'dl': 0.1 if TENSORFLOW_AVAILABLE else 0.0
        }
    
    def _initialize_deep_models(self):
        """Initialize deep learning models"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        # Spread prediction model
        spread_model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(None,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        spread_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['dl_spread'] = spread_model
        
        # Total prediction model
        total_model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(None,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        total_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['dl_total'] = total_model
    
    def _init_database(self):
        """Initialize database for model tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                prediction_type TEXT,
                mae REAL,
                mse REAL,
                r2_score REAL,
                accuracy_pct REAL,
                training_date DATETIME,
                validation_method TEXT
            )
        ''')
        
        # Feature importance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                feature_name TEXT,
                importance_score REAL,
                training_date DATETIME
            )
        ''')
        
        # Prediction history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                model_name TEXT,
                prediction_type TEXT,
                predicted_value REAL,
                actual_value REAL,
                confidence REAL,
                prediction_date DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def train_models(self, training_data: pd.DataFrame, target_columns: Dict[str, str]):
        """Train all models with cross-validation"""
        self.logger.info("Starting model training...")
        
        # Prepare features
        feature_vectors = []
        for _, row in training_data.iterrows():
            features = self.feature_engineer.create_comprehensive_features(
                row.to_dict(), {}, row.to_dict()
            )
            feature_vector = self.feature_engineer.features_to_vector(features)
            feature_vectors.append(feature_vector.flatten())
        
        X = np.array(feature_vectors)
        
        # Train models for each target
        for target_type, target_col in target_columns.items():
            y = training_data[target_col].values
            
            self.logger.info(f"Training {target_type} models...")
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            for model_type in ['xgb', 'rf', 'mlp']:
                model_key = f"{model_type}_{target_type}"
                if model_key not in self.models:
                    continue
                
                self.logger.info(f"Training {model_key}...")
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    self.models[model_key].fit(X_train, y_train)
                    
                    # Validate
                    y_pred = self.models[model_key].predict(X_val)
                    mae = mean_absolute_error(y_val, y_pred)
                    cv_scores.append(mae)
                
                # Final training on full dataset
                self.models[model_key].fit(X, y)
                
                # Store performance metrics
                self._store_model_performance(
                    model_key, target_type, cv_scores, X, y
                )
                
                self.logger.info(f"{model_key} CV MAE: {np.mean(cv_scores):.3f}")
        
        # Train deep learning models if available
        if TENSORFLOW_AVAILABLE:
            self._train_deep_models(X, training_data, target_columns)
        
        # Optimize ensemble weights
        self._optimize_ensemble_weights(X, training_data, target_columns)
        
        self.logger.info("Model training completed!")
    
    def _train_deep_models(self, X: np.ndarray, training_data: pd.DataFrame, 
                          target_columns: Dict[str, str]):
        """Train deep learning models"""
        for target_type, target_col in target_columns.items():
            model_key = f"dl_{target_type}"
            if model_key not in self.models:
                continue
            
            y = training_data[target_col].values
            
            # Split data for training
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train deep model
            self.models[model_key].fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=0
            )
    
    def _optimize_ensemble_weights(self, X: np.ndarray, training_data: pd.DataFrame,
                                 target_columns: Dict[str, str]):
        """Optimize ensemble weights using validation data"""
        # This is a simplified approach - could use more sophisticated optimization
        best_weights = self.ensemble_weights.copy()
        best_score = float('inf')
        
        # Simple grid search for weights
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for xgb_w in weight_options:
            for rf_w in weight_options:
                for mlp_w in weight_options:
                    if xgb_w + rf_w + mlp_w > 1.0:
                        continue
                    
                    dl_w = 1.0 - (xgb_w + rf_w + mlp_w)
                    
                    test_weights = {
                        'xgb': xgb_w,
                        'rf': rf_w, 
                        'mlp': mlp_w,
                        'dl': dl_w
                    }
                    
                    # Evaluate ensemble with these weights
                    score = self._evaluate_ensemble(X, training_data, target_columns, test_weights)
                    
                    if score < best_score:
                        best_score = score
                        best_weights = test_weights
        
        self.ensemble_weights = best_weights
        self.logger.info(f"Optimized ensemble weights: {best_weights}")
    
    def _evaluate_ensemble(self, X: np.ndarray, training_data: pd.DataFrame,
                          target_columns: Dict[str, str], weights: Dict[str, float]) -> float:
        """Evaluate ensemble performance with given weights"""
        total_error = 0
        
        for target_type, target_col in target_columns.items():
            y_true = training_data[target_col].values
            y_pred = self._ensemble_predict(X, target_type, weights)
            mae = mean_absolute_error(y_true, y_pred)
            total_error += mae
        
        return total_error / len(target_columns)
    
    def _ensemble_predict(self, X: np.ndarray, target_type: str, 
                         weights: Dict[str, float]) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        total_weight = 0
        
        for model_type, weight in weights.items():
            model_key = f"{model_type}_{target_type}"
            if model_key in self.models and weight > 0:
                pred = self.models[model_key].predict(X)
                predictions.append(pred * weight)
                total_weight += weight
        
        if predictions:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
            return ensemble_pred
        else:
            return np.zeros(len(X))
    
    def predict(self, team_data: Dict, opponent_data: Dict, 
               game_context: Dict) -> EnsemblePrediction:
        """Make ensemble prediction for a game"""
        
        # Create features
        features = self.feature_engineer.create_comprehensive_features(
            team_data, opponent_data, game_context
        )
        X = self.feature_engineer.features_to_vector(features)
        
        # Get individual model predictions
        individual_predictions = []
        
        for model_type in ['xgb', 'rf', 'mlp', 'dl']:
            if f"{model_type}_spread" in self.models:
                spread_pred = self.models[f"{model_type}_spread"].predict(X)[0]
                total_pred = self.models[f"{model_type}_total"].predict(X)[0]
                
                # Convert spread to win probability
                win_prob = self._spread_to_win_prob(spread_pred)
                
                # Get feature importance (for tree-based models)
                importance = {}
                if hasattr(self.models[f"{model_type}_spread"], 'feature_importances_'):
                    importance = dict(enumerate(
                        self.models[f"{model_type}_spread"].feature_importances_
                    ))
                
                individual_predictions.append(ModelPrediction(
                    model_name=model_type,
                    win_probability=win_prob,
                    point_spread=spread_pred,
                    total_points=total_pred,
                    confidence=0.8,  # Could be model-specific
                    feature_importance=importance
                ))
        
        # Create ensemble prediction
        final_spread = self._ensemble_predict(X, 'spread', self.ensemble_weights)[0]
        final_total = self._ensemble_predict(X, 'total', self.ensemble_weights)[0]
        final_win_prob = self._spread_to_win_prob(final_spread)
        
        # Calculate model agreement
        spreads = [p.point_spread for p in individual_predictions]
        model_agreement = 1.0 / (1.0 + np.std(spreads)) if spreads else 0.5
        
        # Calculate prediction confidence
        prediction_confidence = min(model_agreement * 1.2, 1.0)
        
        return EnsemblePrediction(
            final_win_probability=final_win_prob,
            final_point_spread=final_spread,
            final_total_points=final_total,
            model_agreement=model_agreement,
            prediction_confidence=prediction_confidence,
            individual_predictions=individual_predictions
        )
    
    def _spread_to_win_prob(self, spread: float) -> float:
        """Convert point spread to win probability"""
        # Using logistic function calibrated for basketball
        return 1 / (1 + np.exp(-spread / 3.5))
    
    def _store_model_performance(self, model_name: str, prediction_type: str,
                               cv_scores: List[float], X: np.ndarray, y: np.ndarray):
        """Store model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate metrics on full dataset
        y_pred = self.models[model_name].predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, prediction_type, mae, mse, r2_score, accuracy_pct, training_date, validation_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name, prediction_type, mae, mse, r2, 
            np.mean(cv_scores), datetime.now(), 'TimeSeriesCV'
        ))
        
        conn.commit()
        conn.close()
    
    def save_models(self, model_dir: str = "trained_models"):
        """Save all trained models"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'dl_' in model_name and TENSORFLOW_AVAILABLE:
                # Save TensorFlow model
                model.save(model_path / f"{model_name}.h5")
            else:
                # Save scikit-learn model
                joblib.dump(model, model_path / f"{model_name}.joblib")
        
        # Save ensemble weights
        with open(model_path / "ensemble_weights.json", 'w') as f:
            json.dump(self.ensemble_weights, f)
        
        self.logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "trained_models"):
        """Load trained models"""
        model_path = Path(model_dir)
        
        for model_file in model_path.glob("*"):
            if model_file.suffix == '.h5' and TENSORFLOW_AVAILABLE:
                # Load TensorFlow model
                model_name = model_file.stem
                self.models[model_name] = keras.models.load_model(model_file)
            elif model_file.suffix == '.joblib':
                # Load scikit-learn model
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
        
        # Load ensemble weights
        weights_file = model_path / "ensemble_weights.json"
        if weights_file.exists():
            with open(weights_file, 'r') as f:
                self.ensemble_weights = json.load(f)
        
        self.logger.info(f"Models loaded from {model_dir}")

# Testing function
def test_advanced_pipeline():
    """Test the advanced ML pipeline"""
    pipeline = AdvancedMLPipeline()
    
    print("🤖 Advanced ML Pipeline Demo")
    print("=" * 50)
    
    # Create sample data for testing
    sample_team_data = {
        'offensive_efficiency': 105.2,
        'defensive_efficiency': 98.7,
        'tempo': 72.1,
        'recent_results': [
            {'win': True, 'margin': 12},
            {'win': True, 'margin': 3},
            {'win': False, 'margin': -7}
        ],
        'kenpom_rating': 15.2
    }
    
    sample_opponent_data = {
        'offensive_efficiency': 102.1,
        'defensive_efficiency': 101.2,
        'tempo': 68.9,
        'recent_results': [
            {'win': False, 'margin': -5},
            {'win': True, 'margin': 8},
            {'win': True, 'margin': 15}
        ],
        'kenpom_rating': 22.7
    }
    
    sample_game_context = {
        'tournament_context': 'regular_season',
        'neutral_site': False,
        'is_rivalry': True,
        'venue_capacity': 18000
    }
    
    # Test feature engineering
    features = pipeline.feature_engineer.create_comprehensive_features(
        sample_team_data, sample_opponent_data, sample_game_context
    )
    
    print(f"Generated {len(features.team_efficiency_features)} efficiency features")
    print(f"Generated {len(features.momentum_features)} momentum features")
    print(f"Generated {len(features.advanced_stats_features)} advanced features")
    
    # Test prediction (would need trained models)
    print("\n✅ Advanced ML Pipeline ready for training!")
    print("Features engineered successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_advanced_pipeline()