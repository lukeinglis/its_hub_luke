"""
Example questions for the ITS demo.

These questions are loaded directly from the benchmark datasets used in its_hub:
- MATH500: HuggingFaceH4/MATH-500
- AIME-2024: Maxwell-Jia/AIME_2024
"""

from typing import List, Dict
import datasets
import logging

logger = logging.getLogger(__name__)

# Cache for loaded datasets
_MATH500_CACHE = None
_AIME_CACHE = None


def _load_math500():
    """Load MATH500 dataset and cache it."""
    global _MATH500_CACHE
    if _MATH500_CACHE is None:
        try:
            logger.info("Loading MATH500 dataset from HuggingFace (streaming mode)...")
            # Use streaming to avoid downloading entire dataset
            ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
            # Take first 50 problems and convert to list for caching
            _MATH500_CACHE = list(ds.take(50))
            logger.info(f"Loaded {len(_MATH500_CACHE)} MATH500 problems")
        except Exception as e:
            logger.error(f"Failed to load MATH500: {e}")
            _MATH500_CACHE = []
    return _MATH500_CACHE


def _load_aime_2024():
    """Load AIME-2024 dataset and cache it."""
    global _AIME_CACHE
    if _AIME_CACHE is None:
        try:
            logger.info("Loading AIME-2024 dataset from HuggingFace...")
            # AIME is small (30 problems), so load it all
            ds = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
            _AIME_CACHE = list(ds)
            logger.info(f"Loaded {len(_AIME_CACHE)} AIME-2024 problems")
        except Exception as e:
            logger.error(f"Failed to load AIME-2024: {e}")
            _AIME_CACHE = []
    return _AIME_CACHE


def _categorize_math500_problem(problem: dict) -> str:
    """Extract category from MATH500 problem."""
    # MATH500 has a 'type' field indicating the category
    return problem.get("type", "Math")


def _determine_difficulty(problem: dict, dataset: str) -> str:
    """Determine difficulty level of a problem."""
    if dataset == "AIME":
        # All AIME problems are hard
        return "Hard"
    elif dataset == "MATH500":
        # MATH problems have a 'level' field (1-5)
        level = problem.get("level", 3)
        if isinstance(level, str):
            level = int(level.replace("Level ", ""))

        if level <= 2:
            return "Easy"
        elif level <= 4:
            return "Medium"
        else:
            return "Hard"
    return "Medium"


def _get_best_algorithms_for_problem(difficulty: str, category: str) -> List[str]:
    """Suggest which algorithms work best for this problem type."""
    # All problems benefit from process-based methods
    if difficulty == "Hard":
        return ["beam_search", "particle_filtering", "entropic_particle_filtering", "particle_gibbs"]
    elif difficulty == "Medium":
        return ["beam_search", "particle_filtering", "self_consistency", "best_of_n"]
    else:  # Easy
        return ["self_consistency", "best_of_n", "beam_search"]


def _format_math500_problem(problem: dict, idx: int) -> Dict[str, str]:
    """Format a MATH500 problem for the UI."""
    category = _categorize_math500_problem(problem)
    difficulty = _determine_difficulty(problem, "MATH500")

    return {
        "category": f"MATH500 - {category}",
        "difficulty": difficulty,
        "question": problem["problem"],
        "expected_answer": problem["answer"],
        "best_for": _get_best_algorithms_for_problem(difficulty, category),
        "why": f"MATH500 Level {problem.get('level', '?')} {category} problem",
        "source": "MATH500",
        "source_id": problem.get("unique_id", idx),
    }


def _format_aime_problem(problem: dict, idx: int) -> Dict[str, str]:
    """Format an AIME-2024 problem for the UI."""
    # AIME problems use uppercase keys initially, normalize them
    problem_text = problem.get("Problem") or problem.get("problem", "")
    answer = problem.get("Answer") or problem.get("answer", "")

    # Convert answer to string if it's not already
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "category": "AIME-2024",
        "difficulty": "Hard",
        "question": problem_text,
        "expected_answer": answer,
        "best_for": ["beam_search", "particle_filtering", "entropic_particle_filtering", "particle_gibbs"],
        "why": "AIME competition problem - requires advanced reasoning",
        "source": "AIME-2024",
        "source_id": problem.get("ID") or problem.get("id", idx),
    }


def get_all_questions() -> List[Dict[str, str]]:
    """
    Get all example questions from benchmark datasets.

    Returns a mix of MATH500 and AIME-2024 problems organized by difficulty.
    """
    questions = []

    # Load MATH500 dataset
    math500_ds = _load_math500()
    if math500_ds:
        # Sample problems from different difficulty levels
        try:
            # Get a diverse set: 3 easy, 4 medium, 3 hard from MATH500
            easy_problems = [p for i, p in enumerate(math500_ds)
                           if _determine_difficulty(p, "MATH500") == "Easy"][:3]
            medium_problems = [p for i, p in enumerate(math500_ds)
                             if _determine_difficulty(p, "MATH500") == "Medium"][:4]
            hard_problems = [p for i, p in enumerate(math500_ds)
                           if _determine_difficulty(p, "MATH500") == "Hard"][:3]

            for i, problem in enumerate(easy_problems + medium_problems + hard_problems):
                questions.append(_format_math500_problem(problem, i))
        except Exception as e:
            logger.error(f"Error sampling MATH500 problems: {e}")

    # Load AIME-2024 dataset
    aime_ds = _load_aime_2024()
    if aime_ds:
        # Take first 5 AIME problems
        try:
            for i, problem in enumerate(list(aime_ds)[:5]):
                questions.append(_format_aime_problem(problem, i))
        except Exception as e:
            logger.error(f"Error sampling AIME problems: {e}")

    # If we couldn't load any datasets, provide a fallback
    if not questions:
        logger.warning("Could not load benchmark datasets, using fallback questions")
        questions = _get_fallback_questions()

    return questions


def get_questions_by_algorithm(algorithm: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Get example questions best suited for a specific algorithm.

    Args:
        algorithm: Algorithm name (e.g., 'beam_search', 'best_of_n')
        limit: Maximum number of questions to return

    Returns:
        List of question dictionaries
    """
    all_questions = get_all_questions()

    # Filter questions that are good for this algorithm
    suitable = [q for q in all_questions if algorithm in q["best_for"]]

    # If we don't have enough, add others
    if len(suitable) < limit:
        remaining = [q for q in all_questions if algorithm not in q["best_for"]]
        suitable.extend(remaining[:limit - len(suitable)])

    return suitable[:limit]


def get_questions_by_difficulty(difficulty: str) -> List[Dict[str, str]]:
    """
    Get example questions by difficulty level.

    Args:
        difficulty: 'Easy', 'Medium', or 'Hard'

    Returns:
        List of question dictionaries
    """
    all_questions = get_all_questions()
    return [q for q in all_questions if q["difficulty"] == difficulty]


def _get_fallback_questions() -> List[Dict[str, str]]:
    """
    Fallback questions if datasets cannot be loaded.
    These are simple examples that don't require external datasets.
    """
    return [
        {
            "category": "Algebra",
            "difficulty": "Easy",
            "question": "Solve for x: 2x + 5 = 13",
            "expected_answer": "x = 4",
            "best_for": ["beam_search", "self_consistency", "best_of_n"],
            "why": "Simple algebraic equation",
            "source": "fallback",
            "source_id": "fallback-1",
        },
        {
            "category": "Algebra",
            "difficulty": "Medium",
            "question": "Solve the quadratic equation: x^2 + 5x + 6 = 0",
            "expected_answer": "x = -2 or x = -3",
            "best_for": ["beam_search", "particle_filtering"],
            "why": "Multi-step quadratic solution",
            "source": "fallback",
            "source_id": "fallback-2",
        },
        {
            "category": "Calculus",
            "difficulty": "Medium",
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
            "expected_answer": "f'(x) = 3x^2 + 4x - 5",
            "best_for": ["beam_search", "self_consistency"],
            "why": "Derivative calculation",
            "source": "fallback",
            "source_id": "fallback-3",
        },
    ]
