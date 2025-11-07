#!/usr/bin/env python3
"""
Basketball LLM Ensemble - 5-Model Weighted Voting System
Uses Ollama to run local GGUF models with specialized analysis
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Warning: ollama not installed. Run: pip install ollama")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for each LLM model in the ensemble"""
    name: str
    model_id: str
    weight: float
    specialty: str
    temperature: float = 0.7


@dataclass
class GameAnalysis:
    """Single model's analysis of a game"""
    model_name: str
    prediction: str  # "home" or "away"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    pick_confidence: float  # Extracted confidence score
    response_time: float


@dataclass
class EnsembleResult:
    """Final ensemble prediction"""
    prediction: str  # "home" or "away"
    confidence: float  # Weighted average
    vote_breakdown: Dict[str, int]  # {"home": 3, "away": 2}
    weighted_score: float  # Total weighted score
    model_analyses: List[GameAnalysis]
    total_time: float


class BasketballLLMEnsemble:
    """
    5-Model LLM Ensemble with Weighted Voting

    Models:
    1. Mistral-7B Instruct (4.2) - General reasoning
    2. OpenChat-7B (4.1) - Conversational analysis
    3. Dolphin-Mistral-7B (4.0) - Contrarian views
    4. CodeLlama-7B (4.0) - Analytical reasoning
    5. Neural-Chat-7B (3.9) - Sports insights
    """

    def __init__(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package required. Run: pip install ollama")

        # Define the 5 models with their weights
        self.models = [
            ModelConfig(
                name="Mistral",
                model_id="mistral:7b-instruct",
                weight=4.2,
                specialty="General instruction following and reasoning",
                temperature=0.7
            ),
            ModelConfig(
                name="OpenChat",
                model_id="openchat:7b",
                weight=4.1,
                specialty="Conversational analysis and nuanced reasoning",
                temperature=0.7
            ),
            ModelConfig(
                name="Dolphin",
                model_id="dolphin-mistral:7b",
                weight=4.0,
                specialty="Uncensored analysis and contrarian views",
                temperature=0.8
            ),
            ModelConfig(
                name="CodeLlama",
                model_id="codellama:7b-instruct",
                weight=4.0,
                specialty="Analytical reasoning and structured output",
                temperature=0.6
            ),
            ModelConfig(
                name="NeuralChat",
                model_id="neural-chat:7b",
                weight=3.9,
                specialty="Sports analysis and betting insights",
                temperature=0.7
            )
        ]

        # Normalize weights to sum to 1.0
        total_weight = sum(m.weight for m in self.models)
        for model in self.models:
            model.weight = model.weight / total_weight

        logger.info(f"🤖 LLM Ensemble initialized with {len(self.models)} models")
        for model in self.models:
            logger.info(f"   • {model.name}: {model.weight:.3f} weight - {model.specialty}")

    def check_models_available(self) -> Dict[str, bool]:
        """Check which models are downloaded in Ollama"""
        try:
            available_models = ollama.list()
            model_names = [m['name'] for m in available_models.get('models', [])]

            status = {}
            for model in self.models:
                # Check if model exists (exact or partial match)
                available = any(model.model_id.split(':')[0] in name for name in model_names)
                status[model.name] = available

            return status
        except Exception as e:
            logger.error(f"Error checking models: {e}")
            return {m.name: False for m in self.models}

    def create_game_analysis_prompt(self, home_team: str, away_team: str,
                                   home_stats: Dict = None, away_stats: Dict = None,
                                   additional_context: str = "") -> str:
        """Create a comprehensive prompt for game analysis"""

        prompt = f"""Analyze this college basketball matchup and predict the winner:

HOME: {home_team}
AWAY: {away_team}

"""

        # Add team stats if available
        if home_stats:
            prompt += f"""HOME TEAM STATS:
- Win Rate: {home_stats.get('win_rate', 'N/A')}
- Points Per Game: {home_stats.get('ppg', 'N/A')}
- Offensive Efficiency: {home_stats.get('off_eff', 'N/A')}
- Defensive Efficiency: {home_stats.get('def_eff', 'N/A')}
- Recent Form: {home_stats.get('recent_form', 'N/A')}

"""

        if away_stats:
            prompt += f"""AWAY TEAM STATS:
- Win Rate: {away_stats.get('win_rate', 'N/A')}
- Points Per Game: {away_stats.get('ppg', 'N/A')}
- Offensive Efficiency: {away_stats.get('off_eff', 'N/A')}
- Defensive Efficiency: {away_stats.get('def_eff', 'N/A')}
- Recent Form: {away_stats.get('recent_form', 'N/A')}

"""

        if additional_context:
            prompt += f"""ADDITIONAL CONTEXT:
{additional_context}

"""

        prompt += """Based on this matchup, provide your prediction in this EXACT format:

PREDICTION: [HOME or AWAY]
CONFIDENCE: [0.50 to 1.00]
REASONING: [Your detailed analysis in 2-3 sentences]

Be analytical and consider:
1. Team statistics and efficiency
2. Recent form and momentum
3. Home court advantage
4. Head-to-head history if known
5. Any contextual factors (injuries, rest, etc.)

Your response:"""

        return prompt

    async def get_single_model_analysis(self, model: ModelConfig, prompt: str) -> GameAnalysis:
        """Get analysis from a single model"""
        start_time = time.time()

        try:
            # Call Ollama API
            response = ollama.generate(
                model=model.model_id,
                prompt=prompt,
                options={
                    "temperature": model.temperature,
                    "num_predict": 500  # Max tokens
                }
            )

            response_time = time.time() - start_time
            response_text = response.get('response', '')

            # Parse response
            prediction, confidence, reasoning = self._parse_model_response(response_text)

            return GameAnalysis(
                model_name=model.name,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                pick_confidence=confidence,
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Error with {model.name}: {e}")
            response_time = time.time() - start_time
            return GameAnalysis(
                model_name=model.name,
                prediction="unknown",
                confidence=0.5,
                reasoning=f"Error: {str(e)}",
                pick_confidence=0.5,
                response_time=response_time
            )

    def _parse_model_response(self, response: str) -> tuple:
        """Parse model response to extract prediction, confidence, and reasoning"""
        prediction = "unknown"
        confidence = 0.5
        reasoning = ""

        try:
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()

                if line.startswith("PREDICTION:"):
                    pred_text = line.replace("PREDICTION:", "").strip().upper()
                    if "HOME" in pred_text:
                        prediction = "home"
                    elif "AWAY" in pred_text:
                        prediction = "away"

                elif line.startswith("CONFIDENCE:"):
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    try:
                        confidence = float(conf_text)
                        confidence = max(0.5, min(1.0, confidence))  # Clamp to [0.5, 1.0]
                    except:
                        confidence = 0.5

                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            # If reasoning is still empty, use the whole response
            if not reasoning:
                reasoning = response[:200]

        except Exception as e:
            logger.warning(f"Error parsing response: {e}")

        return prediction, confidence, reasoning

    async def predict_game(self, home_team: str, away_team: str,
                          home_stats: Dict = None, away_stats: Dict = None,
                          additional_context: str = "") -> EnsembleResult:
        """
        Get ensemble prediction for a game

        Returns:
            EnsembleResult with weighted voting outcome
        """
        start_time = time.time()

        # Create prompt
        prompt = self.create_game_analysis_prompt(
            home_team, away_team, home_stats, away_stats, additional_context
        )

        logger.info(f"🏀 Analyzing: {away_team} @ {home_team}")

        # Get predictions from all models (sequentially to avoid overwhelming Ollama)
        model_analyses = []
        for model in self.models:
            logger.info(f"   • Querying {model.name}...")
            analysis = await self.get_single_model_analysis(model, prompt)
            model_analyses.append(analysis)
            logger.info(f"     → {analysis.prediction.upper()} ({analysis.confidence:.2f})")

        # Calculate weighted voting
        result = self._calculate_weighted_vote(model_analyses)
        result.total_time = time.time() - start_time

        logger.info(f"✅ Ensemble Decision: {result.prediction.upper()} "
                   f"({result.confidence:.2f} confidence) in {result.total_time:.1f}s")

        return result

    def _calculate_weighted_vote(self, analyses: List[GameAnalysis]) -> EnsembleResult:
        """Calculate weighted vote from all model analyses"""

        # Count raw votes
        vote_breakdown = {"home": 0, "away": 0, "unknown": 0}
        for analysis in analyses:
            vote_breakdown[analysis.prediction] = vote_breakdown.get(analysis.prediction, 0) + 1

        # Calculate weighted scores
        home_score = 0.0
        away_score = 0.0

        for i, analysis in enumerate(analyses):
            model_weight = self.models[i].weight

            if analysis.prediction == "home":
                home_score += model_weight * analysis.confidence
            elif analysis.prediction == "away":
                away_score += model_weight * analysis.confidence
            else:
                # Unknown - split weight
                home_score += model_weight * 0.5
                away_score += model_weight * 0.5

        # Determine final prediction
        if home_score > away_score:
            prediction = "home"
            confidence = home_score / (home_score + away_score)
            weighted_score = home_score
        else:
            prediction = "away"
            confidence = away_score / (home_score + away_score)
            weighted_score = away_score

        return EnsembleResult(
            prediction=prediction,
            confidence=confidence,
            vote_breakdown=vote_breakdown,
            weighted_score=weighted_score,
            model_analyses=analyses,
            total_time=0.0  # Will be set by caller
        )

    def print_detailed_analysis(self, result: EnsembleResult, home_team: str, away_team: str):
        """Print detailed ensemble analysis"""
        print("\n" + "="*70)
        print(f"🏀 ENSEMBLE ANALYSIS: {away_team} @ {home_team}")
        print("="*70)

        print(f"\n🎯 FINAL PREDICTION: {result.prediction.upper()}")
        print(f"📊 CONFIDENCE: {result.confidence:.1%}")
        print(f"⚖️  WEIGHTED SCORE: {result.weighted_score:.3f}")
        print(f"🗳️  VOTE BREAKDOWN: {result.vote_breakdown}")
        print(f"⏱️  TOTAL TIME: {result.total_time:.1f}s")

        print(f"\n📋 MODEL ANALYSES:")
        print("-"*70)

        for i, analysis in enumerate(result.model_analyses):
            model = self.models[i]
            print(f"\n{i+1}. {model.name} (weight: {model.weight:.3f})")
            print(f"   Prediction: {analysis.prediction.upper()}")
            print(f"   Confidence: {analysis.confidence:.2f}")
            print(f"   Response Time: {analysis.response_time:.1f}s")
            print(f"   Reasoning: {analysis.reasoning[:100]}...")

        print("\n" + "="*70 + "\n")


async def test_ensemble():
    """Test the LLM ensemble"""
    print("🤖 Testing Basketball LLM Ensemble\n")

    ensemble = BasketballLLMEnsemble()

    # Check available models
    print("📋 Checking available models:")
    status = ensemble.check_models_available()
    for model_name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {model_name}")

    if not all(status.values()):
        print("\n⚠️  Some models not available. Run setup script to download them.")
        print("   Command: ./setup_ollama_models.sh")
        return

    # Test prediction
    print("\n🏀 Running test prediction...\n")

    home_stats = {
        "win_rate": "0.750",
        "ppg": "78.5",
        "off_eff": "112.3",
        "def_eff": "98.7",
        "recent_form": "W-W-L-W-W"
    }

    away_stats = {
        "win_rate": "0.600",
        "ppg": "72.1",
        "off_eff": "105.8",
        "def_eff": "102.3",
        "recent_form": "L-W-W-L-W"
    }

    result = await ensemble.predict_game(
        home_team="Duke",
        away_team="North Carolina",
        home_stats=home_stats,
        away_stats=away_stats,
        additional_context="Rivalry game at Cameron Indoor Stadium"
    )

    ensemble.print_detailed_analysis(result, "Duke", "North Carolina")


if __name__ == "__main__":
    asyncio.run(test_ensemble())
