#!/usr/bin/env python3
"""
Basketball Model Training Pipeline
Comprehensive ML training system for college basketball betting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Stores model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    cross_val_score: float = 0.0

@dataclass 
class TrainingConfig:
    """Training configuration parameters."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    grid_search: bool = True
    scale_features: bool = True
    save_models: bool = True
    model_dir: str = "models"

class BasketballFeatureEngine:
    """Generates basketball-specific features for ML training."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_game_features(self, game_data: Dict) -> Dict[str, float]:
        """Create comprehensive game-level features."""
        features = {}
        
        # Basic team stats
        for team in ['home', 'away']:
            prefix = f"{team}_"
            team_data = game_data.get(f'{team}_team', {})
            
            # Offensive features
            features[f"{prefix}ppg"] = team_data.get('points_per_game', 70.0)
            features[f"{prefix}fg_pct"] = team_data.get('field_goal_percentage', 0.45)
            features[f"{prefix}three_pct"] = team_data.get('three_point_percentage', 0.35)
            features[f"{prefix}ft_pct"] = team_data.get('free_throw_percentage', 0.75)
            features[f"{prefix}assists_pg"] = team_data.get('assists_per_game', 15.0)
            features[f"{prefix}turnovers_pg"] = team_data.get('turnovers_per_game', 12.0)
            
            # Defensive features
            features[f"{prefix}opp_ppg"] = team_data.get('opponent_points_per_game', 70.0)
            features[f"{prefix}steals_pg"] = team_data.get('steals_per_game', 7.0)
            features[f"{prefix}blocks_pg"] = team_data.get('blocks_per_game', 4.0)
            features[f"{prefix}def_rebounds_pg"] = team_data.get('def_rebounds_per_game', 25.0)
            
            # Advanced metrics
            features[f"{prefix}pace"] = team_data.get('pace', 70.0)
            features[f"{prefix}offensive_efficiency"] = team_data.get('offensive_efficiency', 1.1)
            features[f"{prefix}defensive_efficiency"] = team_data.get('defensive_efficiency', 1.0)
            features[f"{prefix}net_efficiency"] = team_data.get('net_efficiency', 0.1)
            
            # Recent form
            features[f"{prefix}last5_wins"] = team_data.get('last_5_wins', 3)
            features[f"{prefix}home_record"] = team_data.get('home_win_pct', 0.6) if team == 'home' else team_data.get('away_win_pct', 0.4)
            features[f"{prefix}conference_record"] = team_data.get('conference_win_pct', 0.5)
            
            # Rankings and strength
            features[f"{prefix}kenpom_rank"] = team_data.get('kenpom_ranking', 150)
            features[f"{prefix}rpi_rank"] = team_data.get('rpi_ranking', 150)
            features[f"{prefix}sos"] = team_data.get('strength_of_schedule', 0.5)
            
        # Matchup-specific features
        features['pace_differential'] = features.get('home_pace', 70) - features.get('away_pace', 70)
        features['offensive_advantage_home'] = features.get('home_offensive_efficiency', 1.1) - features.get('away_defensive_efficiency', 1.0)
        features['offensive_advantage_away'] = features.get('away_offensive_efficiency', 1.1) - features.get('home_defensive_efficiency', 1.0)
        features['experience_edge'] = features.get('home_kenpom_rank', 150) - features.get('away_kenpom_rank', 150)
        
        # Situational features
        features['days_rest_home'] = game_data.get('home_days_rest', 2)
        features['days_rest_away'] = game_data.get('away_days_rest', 2)
        features['is_conference_game'] = 1.0 if game_data.get('is_conference_game') else 0.0
        features['is_tournament'] = 1.0 if game_data.get('is_tournament') else 0.0
        features['is_neutral_site'] = 1.0 if game_data.get('is_neutral_site') else 0.0
        
        # Betting market features
        features['spread'] = game_data.get('spread', 0.0)
        features['total'] = game_data.get('total', 140.0)
        features['home_moneyline'] = game_data.get('home_moneyline', -110)
        features['away_moneyline'] = game_data.get('away_moneyline', -110)
        
        return features
        
    def create_injury_features(self, injury_data: Dict) -> Dict[str, float]:
        """Create injury-related features."""
        features = {}
        
        for team in ['home', 'away']:
            team_injuries = injury_data.get(f'{team}_injuries', [])
            
            features[f"{team}_injury_count"] = len(team_injuries)
            features[f"{team}_starter_injuries"] = sum(1 for inj in team_injuries if inj.get('is_starter'))
            features[f"{team}_key_player_out"] = 1.0 if any(inj.get('importance', 0) > 0.7 for inj in team_injuries) else 0.0
            features[f"{team}_depth_impact"] = sum(inj.get('depth_impact', 0) for inj in team_injuries) / 5.0
            
        return features
        
    def create_tournament_features(self, tournament_data: Dict) -> Dict[str, float]:
        """Create March Madness tournament-specific features."""
        features = {}
        
        if not tournament_data.get('is_tournament'):
            return {f'tournament_{key}': 0.0 for key in ['seed_diff', 'experience_home', 'experience_away', 
                                                        'upset_probability', 'coaching_factor']}
        
        features['tournament_seed_diff'] = tournament_data.get('seed_differential', 0)
        features['tournament_experience_home'] = tournament_data.get('home_tournament_experience', 0)
        features['tournament_experience_away'] = tournament_data.get('away_tournament_experience', 0)
        features['tournament_upset_probability'] = tournament_data.get('upset_probability', 0.5)
        features['tournament_coaching_factor'] = tournament_data.get('coaching_experience_factor', 1.0)
        
        return features

class BasketballModelTrainer:
    """Main training pipeline for college basketball betting models."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.feature_engine = BasketballFeatureEngine()
        self.models = {}
        self.metrics = {}
        self.scalers = {}
        
        # Model configurations
        self.model_configs = {
            'spread_classifier': {
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.config.random_state),
                    'gradient_boosting': GradientBoostingClassifier(random_state=self.config.random_state),
                    'logistic_regression': LogisticRegression(random_state=self.config.random_state),
                    'svm': SVC(probability=True, random_state=self.config.random_state),
                    'neural_network': MLPClassifier(random_state=self.config.random_state)
                },
                'grid_params': {
                    'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
                    'gradient_boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
                    'logistic_regression': {'C': [0.1, 1, 10]},
                    'svm': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
                    'neural_network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
                }
            },
            'total_regressor': {
                'models': {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
                    'gradient_boosting': GradientBoostingClassifier(random_state=self.config.random_state),
                    'linear_regression': LinearRegression(),
                    'svr': SVR(),
                    'neural_network': MLPRegressor(random_state=self.config.random_state)
                },
                'grid_params': {
                    'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
                    'linear_regression': {},
                    'svr': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
                    'neural_network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
                }
            },
            'upset_classifier': {
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=200, random_state=self.config.random_state),
                    'gradient_boosting': GradientBoostingClassifier(random_state=self.config.random_state),
                    'logistic_regression': LogisticRegression(random_state=self.config.random_state),
                    'neural_network': MLPClassifier(random_state=self.config.random_state, max_iter=500)
                },
                'grid_params': {
                    'random_forest': {'n_estimators': [100, 200, 300], 'max_depth': [15, 25, None]},
                    'gradient_boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.15]},
                    'logistic_regression': {'C': [0.1, 1, 10, 100]},
                    'neural_network': {'hidden_layer_sizes': [(100,), (100, 50), (150, 75)], 'alpha': [0.0001, 0.001]}
                }
            }
        }
        
    def prepare_features(self, game_data: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare feature matrix from game data."""
        logger.info(f"Preparing features for {len(game_data)} games")
        
        feature_rows = []
        targets = {'spread_result': [], 'total_result': [], 'upset_result': []}
        
        for game in game_data:
            # Extract features
            game_features = self.feature_engine.create_game_features(game)
            injury_features = self.feature_engine.create_injury_features(game.get('injuries', {}))
            tournament_features = self.feature_engine.create_tournament_features(game.get('tournament', {}))
            
            # Combine all features
            all_features = {**game_features, **injury_features, **tournament_features}
            feature_rows.append(all_features)
            
            # Extract targets
            result = game.get('result', {})
            targets['spread_result'].append(1 if result.get('home_covered_spread') else 0)
            targets['total_result'].append(result.get('total_points', 140))
            targets['upset_result'].append(1 if result.get('is_upset') else 0)
        
        features_df = pd.DataFrame(feature_rows)
        target_series = {key: pd.Series(values) for key, values in targets.items()}
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        logger.info(f"Feature matrix shape: {features_df.shape}")
        logger.info(f"Features: {list(features_df.columns)}")
        
        return features_df, target_series
        
    def train_model_type(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
        """Train all models for a specific prediction type."""
        logger.info(f"Training {model_type} models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y if model_type.endswith('classifier') else None
        )
        
        # Scale features if needed
        if self.config.scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_type] = scaler
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        model_results = {}
        config = self.model_configs[model_type]
        
        for model_name, model in config['models'].items():
            logger.info(f"Training {model_name} for {model_type}")
            
            try:
                # Grid search if enabled
                if self.config.grid_search and model_name in config['grid_params']:
                    grid_search = GridSearchCV(
                        model, config['grid_params'][model_name], 
                        cv=self.config.cv_folds, scoring='roc_auc' if model_type.endswith('classifier') else 'neg_mean_squared_error'
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = best_model.predict(X_test_scaled)
                if model_type.endswith('classifier'):
                    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = ModelMetrics()
                
                if model_type.endswith('classifier'):
                    metrics.accuracy = accuracy_score(y_test, y_pred)
                    metrics.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics.roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    # Cross validation
                    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=self.config.cv_folds, scoring='roc_auc')
                    metrics.cross_val_score = cv_scores.mean()
                    
                else:  # Regressor
                    metrics.mse = mean_squared_error(y_test, y_pred)
                    metrics.mae = mean_absolute_error(y_test, y_pred)
                    metrics.r2_score = r2_score(y_test, y_pred)
                    
                    # Cross validation
                    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=self.config.cv_folds, scoring='neg_mean_squared_error')
                    metrics.cross_val_score = -cv_scores.mean()
                
                model_results[model_name] = {
                    'model': best_model,
                    'metrics': metrics,
                    'feature_importance': self._get_feature_importance(best_model, X.columns) if hasattr(best_model, 'feature_importances_') else None
                }
                
                logger.info(f"{model_name} trained successfully")
                if model_type.endswith('classifier'):
                    logger.info(f"Accuracy: {metrics.accuracy:.3f}, ROC-AUC: {metrics.roc_auc:.3f}")
                else:
                    logger.info(f"MSE: {metrics.mse:.3f}, R²: {metrics.r2_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return model_results
        
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return None
                
            return dict(zip(feature_names, importance))
        except:
            return None
            
    def train_all_models(self, game_data: List[Dict]) -> Dict[str, Any]:
        """Train all basketball betting models."""
        logger.info("Starting comprehensive model training")
        
        # Prepare features
        X, targets = self.prepare_features(game_data)
        
        results = {}
        
        # Train each model type
        for model_type in ['spread_classifier', 'total_regressor', 'upset_classifier']:
            target_name = model_type.replace('_classifier', '_result').replace('_regressor', '_result')
            y = targets[target_name]
            
            model_results = self.train_model_type(X, y, model_type)
            results[model_type] = model_results
            
            # Select best model
            if model_type.endswith('classifier'):
                best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['metrics'].roc_auc)
            else:
                best_model_name = min(model_results.keys(), key=lambda k: model_results[k]['metrics'].mse)
            
            results[model_type]['best_model'] = best_model_name
            self.models[model_type] = model_results[best_model_name]['model']
            self.metrics[model_type] = model_results[best_model_name]['metrics']
            
            logger.info(f"Best {model_type} model: {best_model_name}")
        
        # Save models if configured
        if self.config.save_models:
            self.save_models()
        
        return results
        
    def save_models(self):
        """Save trained models to disk."""
        import os
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save models
        for model_type, model in self.models.items():
            model_path = f"{self.config.model_dir}/{model_type}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_type} model to {model_path}")
        
        # Save scalers
        for scaler_type, scaler in self.scalers.items():
            scaler_path = f"{self.config.model_dir}/{scaler_type}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {scaler_type} scaler to {scaler_path}")
        
        # Save metrics
        metrics_dict = {model_type: vars(metrics) for model_type, metrics in self.metrics.items()}
        metrics_path = f"{self.config.model_dir}/model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def load_models(self):
        """Load trained models from disk."""
        for model_type in ['spread_classifier', 'total_regressor', 'upset_classifier']:
            model_path = f"{self.config.model_dir}/{model_type}_model.joblib"
            try:
                self.models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load scalers
        for scaler_type in ['spread_classifier', 'total_regressor', 'upset_classifier']:
            scaler_path = f"{self.config.model_dir}/{scaler_type}_scaler.joblib"
            try:
                self.scalers[scaler_type] = joblib.load(scaler_path)
                logger.info(f"Loaded {scaler_type} scaler from {scaler_path}")
            except FileNotFoundError:
                logger.warning(f"Scaler file not found: {scaler_path}")
    
    def evaluate_models(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate trained models on test data."""
        logger.info("Evaluating models on test data")
        
        X_test, y_test = self.prepare_features(test_data)
        evaluation_results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Evaluating {model_type}")
            
            # Scale features if needed
            if model_type in self.scalers:
                X_test_scaled = self.scalers[model_type].transform(X_test)
            else:
                X_test_scaled = X_test
            
            target_name = model_type.replace('_classifier', '_result').replace('_regressor', '_result')
            y_true = y_test[target_name]
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = ModelMetrics()
            
            if model_type.endswith('classifier'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                metrics.accuracy = accuracy_score(y_true, y_pred)
                metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.roc_auc = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics.mse = mean_squared_error(y_true, y_pred)
                metrics.mae = mean_absolute_error(y_true, y_pred)
                metrics.r2_score = r2_score(y_true, y_pred)
            
            evaluation_results[model_type] = metrics
            
            logger.info(f"{model_type} evaluation complete")
            if model_type.endswith('classifier'):
                logger.info(f"Test Accuracy: {metrics.accuracy:.3f}, Test ROC-AUC: {metrics.roc_auc:.3f}")
            else:
                logger.info(f"Test MSE: {metrics.mse:.3f}, Test R²: {metrics.r2_score:.3f}")
        
        return evaluation_results
    
    def predict(self, game_data: Dict) -> Dict[str, Any]:
        """Make predictions for a single game."""
        # Prepare features
        game_features = self.feature_engine.create_game_features(game_data)
        injury_features = self.feature_engine.create_injury_features(game_data.get('injuries', {}))
        tournament_features = self.feature_engine.create_tournament_features(game_data.get('tournament', {}))
        
        all_features = {**game_features, **injury_features, **tournament_features}
        X = pd.DataFrame([all_features])
        X = X.fillna(X.median())
        
        predictions = {}
        
        for model_type, model in self.models.items():
            # Scale features if needed
            if model_type in self.scalers:
                X_scaled = self.scalers[model_type].transform(X)
            else:
                X_scaled = X
            
            if model_type.endswith('classifier'):
                pred = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0, 1]
                predictions[model_type] = {'prediction': pred, 'probability': pred_proba}
            else:
                pred = model.predict(X_scaled)[0]
                predictions[model_type] = {'prediction': pred}
        
        return predictions

def main():
    """Demo the basketball training pipeline."""
    print("College Basketball Model Training Pipeline")
    print("=" * 50)
    
    # Sample training data (normally would load from database/files)
    sample_games = [
        {
            'home_team': {'points_per_game': 75, 'field_goal_percentage': 0.47, 'pace': 72, 'kenpom_ranking': 25},
            'away_team': {'points_per_game': 72, 'field_goal_percentage': 0.44, 'pace': 68, 'kenpom_ranking': 45},
            'spread': -3.5, 'total': 147, 'is_conference_game': True,
            'result': {'home_covered_spread': True, 'total_points': 149, 'is_upset': False}
        },
        {
            'home_team': {'points_per_game': 68, 'field_goal_percentage': 0.42, 'pace': 65, 'kenpom_ranking': 85},
            'away_team': {'points_per_game': 78, 'field_goal_percentage': 0.49, 'pace': 74, 'kenpom_ranking': 15},
            'spread': 8.5, 'total': 142, 'is_conference_game': False,
            'result': {'home_covered_spread': False, 'total_points': 138, 'is_upset': True}
        }
        # In practice, would have thousands of games
    ]
    
    # Initialize trainer
    config = TrainingConfig(save_models=False, grid_search=False)  # Simplified for demo
    trainer = BasketballModelTrainer(config)
    
    try:
        # Train models (would need more data for real training)
        print("Training models (demo with minimal data)...")
        results = trainer.train_all_models(sample_games * 50)  # Duplicate for demo
        
        print("\nTraining completed successfully!")
        print("Model performance summary:")
        for model_type, metrics in trainer.metrics.items():
            print(f"\n{model_type}:")
            if hasattr(metrics, 'accuracy'):
                print(f"  Accuracy: {metrics.accuracy:.3f}")
                print(f"  ROC-AUC: {metrics.roc_auc:.3f}")
            else:
                print(f"  MSE: {metrics.mse:.3f}")
                print(f"  R²: {metrics.r2_score:.3f}")
        
        # Demo prediction
        print("\n" + "="*30)
        print("Sample Prediction:")
        sample_game = {
            'home_team': {'points_per_game': 80, 'field_goal_percentage': 0.48, 'pace': 75, 'kenpom_ranking': 12},
            'away_team': {'points_per_game': 76, 'field_goal_percentage': 0.46, 'pace': 71, 'kenpom_ranking': 28},
            'spread': -4.5, 'total': 156, 'is_conference_game': True
        }
        
        predictions = trainer.predict(sample_game)
        print(f"Spread prediction: {'Home covers' if predictions['spread_classifier']['probability'] > 0.5 else 'Away covers'}")
        print(f"Confidence: {predictions['spread_classifier']['probability']:.3f}")
        print(f"Predicted total: {predictions['total_regressor']['prediction']:.1f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()