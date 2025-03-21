from typing import Dict, Any, List
from .base_agent import BaseAgent

class StatisticianAgent(BaseAgent):
    """Agent specialized in statistical analysis for medical research."""
    
    def __init__(self, name: str = "Statistical Analyst"):
        super().__init__(name, expertise=["statistics", "data analysis", "hypothesis testing", "p-values"])
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process statistics-related tasks."""
        task_type = task.get("type", "")
        
        if task_type == "suggest_analysis":
            return self._suggest_analysis_method(task)
        elif task_type == "calculate_sample_size":
            return self._calculate_sample_size(task)
        elif task_type == "interpret_results":
            return self._interpret_statistical_results(task)
        else:
            return {"error": "Unknown task type for Statistician Agent"}
    
    def _suggest_analysis_method(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest appropriate statistical methods for a given research question."""
        question = task.get("question", "")
        data_type = task.get("data_type", "unknown")
        outcome_type = task.get("outcome_type", "unknown")
        
        # Logic to suggest statistical methods based on data characteristics
        suggested_methods = []
        
        if outcome_type == "continuous":
            suggested_methods.extend(["t-test", "ANOVA", "linear regression"])
        elif outcome_type == "binary":
            suggested_methods.extend(["logistic regression", "chi-square test"])
        elif outcome_type == "time-to-event":
            suggested_methods.extend(["survival analysis", "Cox proportional hazards"])
        
        return {
            "question": question,
            "data_type": data_type,
            "outcome_type": outcome_type,
            "suggested_methods": suggested_methods,
            "explanation": "These methods are appropriate based on your outcome type and research question."
        }
    
    def _calculate_sample_size(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate required sample size based on statistical parameters."""
        effect_size = task.get("effect_size", 0.5)
        alpha = task.get("alpha", 0.05)
        power = task.get("power", 0.8)
        
        # Simplified sample size calculation logic
        # In a real implementation, this would use proper statistical formulas
        sample_size = int(100 / effect_size)
        
        return {
            "parameters": {
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power
            },
            "required_sample_size": sample_size,
            "notes": "This is a simplified calculation. For a formal study, consult with a biostatistician."
        }
    
    def _interpret_statistical_results(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret statistical results in plain language."""
        p_value = task.get("p_value", 1.0)
        effect_estimate = task.get("effect_estimate", 0)
        confidence_interval = task.get("confidence_interval", [0, 0])
        
        is_significant = p_value < 0.05
        
        interpretation = ""
        if is_significant:
            interpretation = f"The result is statistically significant (p={p_value}), suggesting a real effect."
        else:
            interpretation = f"The result is not statistically significant (p={p_value}), which may indicate no effect or insufficient power."
        
        return {
            "results": {
                "p_value": p_value,
                "effect_estimate": effect_estimate,
                "confidence_interval": confidence_interval
            },
            "is_significant": is_significant,
            "interpretation": interpretation,
            "recommendations": [
                "Consider clinical significance in addition to statistical significance",
                "Evaluate the width of confidence intervals to assess precision"
            ]
        }
