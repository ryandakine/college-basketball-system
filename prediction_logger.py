#!/usr/bin/env python3
"""
Prediction Logger
=================

Logs all predictions to JSON file for tracking performance and calibration.
Saves after every pick with timestamp, confidence, prediction details, and outcome tracking.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from filelock import FileLock
import os


@dataclass
class PredictionLog:
    """Individual prediction log entry"""
    prediction_id: str
    timestamp: str
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    
    # Prediction details
    bet_type: str  # "spread", "total", "moneyline"
    selection: str
    predicted_probability: float
    raw_confidence: float
    calibrated_confidence: float
    
    # Betting info
    line: Optional[float]
    odds: float
    edge_percentage: float
    suggested_units: float
    recommendation: str  # "BET", "STRONG_BET", "SMALL_BET", "AVOID"
    
    # Model info
    model_version: str
    confidence_tier: str
    data_quality_score: float
    model_agreement: float
    
    # Key factors
    key_factors: list
    risk_factors: list
    reasoning: str
    
    # Outcome tracking (filled in after game)
    actual_result: Optional[str] = None
    bet_outcome: Optional[str] = None  # "WIN", "LOSS", "PUSH"
    profit_loss: Optional[float] = None
    outcome_timestamp: Optional[str] = None
    

class PredictionLogger:
    """Handles logging predictions to JSON file"""
    
    def __init__(self, log_path: str = "data/predictions/prediction_log.json"):
        self.log_path = Path(log_path)
        self.lock_path = Path(str(log_path) + ".lock")
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_path.exists():
            self._initialize_log_file()
            
    def _initialize_log_file(self):
        """Create initial log file structure"""
        initial_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Prediction tracking log for college basketball system"
            },
            "predictions": []
        }
        
        with open(self.log_path, 'w') as f:
            json.dump(initial_data, f, indent=2)
            
        self.logger.info(f"Initialized prediction log at {self.log_path}")
        
    def log_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Log a prediction to the JSON file
        
        Args:
            prediction_data: Dictionary containing prediction details
            
        Returns:
            prediction_id: Unique ID for this prediction
        """
        try:
            # Generate prediction ID
            prediction_id = f"{prediction_data['game_id']}_{prediction_data['bet_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create prediction log entry
            log_entry = PredictionLog(
                prediction_id=prediction_id,
                timestamp=datetime.now().isoformat(),
                game_id=prediction_data.get('game_id', ''),
                game_date=prediction_data.get('game_date', ''),
                home_team=prediction_data.get('home_team', ''),
                away_team=prediction_data.get('away_team', ''),
                bet_type=prediction_data.get('bet_type', ''),
                selection=prediction_data.get('selection', ''),
                predicted_probability=prediction_data.get('predicted_probability', 0.0),
                raw_confidence=prediction_data.get('raw_confidence', 0.0),
                calibrated_confidence=prediction_data.get('calibrated_confidence', 0.0),
                line=prediction_data.get('line'),
                odds=prediction_data.get('odds', 0.0),
                edge_percentage=prediction_data.get('edge_percentage', 0.0),
                suggested_units=prediction_data.get('suggested_units', 0.0),
                recommendation=prediction_data.get('recommendation', 'AVOID'),
                model_version=prediction_data.get('model_version', '1.0.0'),
                confidence_tier=prediction_data.get('confidence_tier', 'LOW'),
                data_quality_score=prediction_data.get('data_quality_score', 0.0),
                model_agreement=prediction_data.get('model_agreement', 0.0),
                key_factors=prediction_data.get('key_factors', []),
                risk_factors=prediction_data.get('risk_factors', []),
                reasoning=prediction_data.get('reasoning', '')
            )
            
            # Use file lock to prevent concurrent write issues
            lock = FileLock(self.lock_path, timeout=10)
            
            with lock:
                # Read current log
                with open(self.log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Add new prediction
                log_data['predictions'].append(asdict(log_entry))
                
                # Write back to file
                with open(self.log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                    
            self.logger.info(f"Logged prediction {prediction_id}")
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Error logging prediction: {e}")
            return ""
            
    def update_outcome(self, prediction_id: str, outcome_data: Dict[str, Any]):
        """
        Update prediction with actual outcome after game completes
        
        Args:
            prediction_id: ID of prediction to update
            outcome_data: Dictionary with actual results
        """
        try:
            lock = FileLock(self.lock_path, timeout=10)
            
            with lock:
                # Read current log
                with open(self.log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Find and update prediction
                for prediction in log_data['predictions']:
                    if prediction['prediction_id'] == prediction_id:
                        prediction['actual_result'] = outcome_data.get('actual_result')
                        prediction['bet_outcome'] = outcome_data.get('bet_outcome')
                        prediction['profit_loss'] = outcome_data.get('profit_loss')
                        prediction['outcome_timestamp'] = datetime.now().isoformat()
                        break
                        
                # Write back to file
                with open(self.log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                    
            self.logger.info(f"Updated outcome for prediction {prediction_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating prediction outcome: {e}")
            
    def get_predictions(self, filters: Optional[Dict] = None) -> list:
        """
        Retrieve predictions with optional filtering
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching predictions
        """
        try:
            with open(self.log_path, 'r') as f:
                log_data = json.load(f)
                
            predictions = log_data['predictions']
            
            # Apply filters if provided
            if filters:
                if 'date_from' in filters:
                    predictions = [p for p in predictions if p['game_date'] >= filters['date_from']]
                if 'date_to' in filters:
                    predictions = [p for p in predictions if p['game_date'] <= filters['date_to']]
                if 'bet_type' in filters:
                    predictions = [p for p in predictions if p['bet_type'] == filters['bet_type']]
                if 'recommendation' in filters:
                    predictions = [p for p in predictions if p['recommendation'] == filters['recommendation']]
                    
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return []
            
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics from logged predictions"""
        try:
            predictions = self.get_predictions()
            
            # Filter completed predictions
            completed = [p for p in predictions if p.get('bet_outcome') is not None]
            
            if not completed:
                return {
                    'total_predictions': len(predictions),
                    'completed': 0,
                    'pending': len(predictions)
                }
                
            # Calculate stats
            wins = sum(1 for p in completed if p['bet_outcome'] == 'WIN')
            losses = sum(1 for p in completed if p['bet_outcome'] == 'LOSS')
            pushes = sum(1 for p in completed if p['bet_outcome'] == 'PUSH')
            
            total_profit = sum(p.get('profit_loss', 0.0) for p in completed)
            total_units_bet = sum(p.get('suggested_units', 0.0) for p in completed)
            
            # Win rate by confidence tier
            high_conf = [p for p in completed if p['confidence_tier'] == 'HIGH']
            med_conf = [p for p in completed if p['confidence_tier'] == 'MEDIUM']
            low_conf = [p for p in completed if p['confidence_tier'] == 'LOW']
            
            stats = {
                'total_predictions': len(predictions),
                'completed': len(completed),
                'pending': len(predictions) - len(completed),
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'win_rate': wins / len(completed) if completed else 0.0,
                'total_profit': total_profit,
                'total_units_bet': total_units_bet,
                'roi': (total_profit / total_units_bet * 100) if total_units_bet > 0 else 0.0,
                'by_confidence': {
                    'HIGH': {
                        'count': len(high_conf),
                        'wins': sum(1 for p in high_conf if p['bet_outcome'] == 'WIN'),
                        'win_rate': sum(1 for p in high_conf if p['bet_outcome'] == 'WIN') / len(high_conf) if high_conf else 0.0
                    },
                    'MEDIUM': {
                        'count': len(med_conf),
                        'wins': sum(1 for p in med_conf if p['bet_outcome'] == 'WIN'),
                        'win_rate': sum(1 for p in med_conf if p['bet_outcome'] == 'WIN') / len(med_conf) if med_conf else 0.0
                    },
                    'LOW': {
                        'count': len(low_conf),
                        'wins': sum(1 for p in low_conf if p['bet_outcome'] == 'WIN'),
                        'win_rate': sum(1 for p in low_conf if p['bet_outcome'] == 'WIN') / len(low_conf) if low_conf else 0.0
                    }
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating performance stats: {e}")
            return {}


if __name__ == "__main__":
    # Test the prediction logger
    logging.basicConfig(level=logging.INFO)
    
    logger = PredictionLogger()
    
    # Test prediction
    test_prediction = {
        'game_id': 'test_game_001',
        'game_date': '2025-11-09',
        'home_team': 'Duke',
        'away_team': 'UNC',
        'bet_type': 'spread',
        'selection': 'Duke -3.5',
        'predicted_probability': 0.58,
        'raw_confidence': 0.75,
        'calibrated_confidence': 0.75,
        'line': -3.5,
        'odds': -110,
        'edge_percentage': 0.05,
        'suggested_units': 1.0,
        'recommendation': 'BET',
        'model_version': '1.0.0',
        'confidence_tier': 'MEDIUM',
        'data_quality_score': 0.8,
        'model_agreement': 0.7,
        'key_factors': ['Home court advantage', 'Conference strength'],
        'risk_factors': [],
        'reasoning': 'Strong home team with conference advantage'
    }
    
    prediction_id = logger.log_prediction(test_prediction)
    print(f"✅ Logged prediction: {prediction_id}")
    
    # Test outcome update
    outcome = {
        'actual_result': 'Duke won by 5',
        'bet_outcome': 'WIN',
        'profit_loss': 0.91
    }
    
    logger.update_outcome(prediction_id, outcome)
    print(f"✅ Updated outcome for: {prediction_id}")
    
    # Get stats
    stats = logger.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(json.dumps(stats, indent=2))
