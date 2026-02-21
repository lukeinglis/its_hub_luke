"""
Example questions for the ITS demo.

These questions are curated to demonstrate where ITS algorithms
provide clear improvements over baseline inference. Questions are
inspired by the MATH500 and AIME datasets used in its_hub benchmarks.
"""

from typing import List, Dict

# Example questions organized by difficulty and type
EXAMPLE_QUESTIONS: List[Dict[str, str]] = [
    # ===== EASY: Good for all algorithms =====
    {
        "category": "Basic Math",
        "difficulty": "Easy",
        "question": "What is 2+2? Show your work.",
        "expected_answer": "4",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Simple verification task where self-consistency excels"
    },
    {
        "category": "Algebra",
        "difficulty": "Easy",
        "question": "Solve the equation: 2x + 5 = 13. Show your steps.",
        "expected_answer": "x = 4",
        "best_for": ["beam_search", "particle_filtering"],
        "why": "Clear step-by-step solution path"
    },
    {
        "category": "Logic",
        "difficulty": "Easy",
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "expected_answer": "No, we cannot conclude that. The statement only tells us that some flowers (not necessarily roses) fade quickly. The roses could be among the flowers that don't fade quickly.",
        "best_for": ["best_of_n", "self_consistency"],
        "why": "Logical reasoning with clear correct answer"
    },

    # ===== MEDIUM: Shows clear ITS benefits =====
    {
        "category": "Algebra",
        "difficulty": "Medium",
        "question": "Solve the quadratic equation: x^2 + 5x + 6 = 0. Show all steps and verify your answer.",
        "expected_answer": "x = -2 or x = -3 (factoring: (x+2)(x+3) = 0)",
        "best_for": ["beam_search", "particle_filtering"],
        "why": "Multi-step solution with verification"
    },
    {
        "category": "Calculus",
        "difficulty": "Medium",
        "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3. Show your work step by step.",
        "expected_answer": "f'(x) = 3x^2 + 4x - 5",
        "best_for": ["beam_search", "self_consistency"],
        "why": "Systematic application of derivative rules"
    },
    {
        "category": "Number Theory",
        "difficulty": "Medium",
        "question": "What is the sum of all prime numbers between 10 and 30? List the primes first, then calculate.",
        "expected_answer": "Primes: 11, 13, 17, 19, 23, 29. Sum = 112",
        "best_for": ["particle_filtering", "beam_search"],
        "why": "Requires careful enumeration and calculation"
    },
    {
        "category": "Word Problem",
        "difficulty": "Medium",
        "question": "A train travels 120 miles in 2 hours. If it continues at the same speed, how long will it take to travel 300 miles? Show your reasoning.",
        "expected_answer": "5 hours (speed = 60 mph, time = 300/60 = 5 hours)",
        "best_for": ["best_of_n", "beam_search"],
        "why": "Multi-step word problem"
    },
    {
        "category": "Geometry",
        "difficulty": "Medium",
        "question": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area and perimeter? Show your calculations.",
        "expected_answer": "Area = 60 cm², Perimeter = 34 cm",
        "best_for": ["self_consistency", "beam_search"],
        "why": "Multiple calculations with clear answers"
    },

    # ===== HARD: Best for advanced algorithms =====
    {
        "category": "Algebra",
        "difficulty": "Hard",
        "question": "Let a be a positive real number such that all the roots of x^3 + ax^2 + ax + 1 = 0 are real. Find the smallest possible value of a.",
        "expected_answer": "a = 3 (this is the minimum value for which all roots are real)",
        "best_for": ["particle_gibbs", "entropic_particle_filtering"],
        "why": "Complex optimization requiring multiple refinement passes"
    },
    {
        "category": "Number Theory",
        "difficulty": "Hard",
        "question": "Find all integer solutions to the equation x^2 - 4y^2 = 13. Show your complete reasoning.",
        "expected_answer": "(x, y) = (±7, ±3) are the only integer solutions. This can be factored as (x-2y)(x+2y) = 13.",
        "best_for": ["beam_search", "particle_filtering"],
        "why": "Requires systematic search through possibilities"
    },
    {
        "category": "Combinatorics",
        "difficulty": "Hard",
        "question": "In how many ways can you arrange the letters in the word MATHEMATICS? Show your step-by-step calculation.",
        "expected_answer": "4,989,600 ways (11!/(2!×2!×2!) since M, A, and T each appear twice)",
        "best_for": ["particle_filtering", "beam_search"],
        "why": "Multi-step counting with careful attention to duplicates"
    },
    {
        "category": "Algebra",
        "difficulty": "Hard",
        "question": "If f(x) = x^2 + 3x + 2 and g(x) = 2x - 1, find (f ∘ g)(x) and determine its range when x is in [0, 5].",
        "expected_answer": "(f ∘ g)(x) = 4x^2 + 2x; Range for x ∈ [0,5] is [0, 110]",
        "best_for": ["entropic_particle_filtering", "particle_gibbs"],
        "why": "Function composition requiring careful algebraic manipulation"
    },
    {
        "category": "Logic Puzzle",
        "difficulty": "Hard",
        "question": "Three friends (Alice, Bob, and Carol) have different favorite colors (red, blue, green). Alice doesn't like red. Bob's favorite color is not blue. Carol likes green. What color does each person like?",
        "expected_answer": "Carol: green, Alice: blue, Bob: red",
        "best_for": ["beam_search", "particle_filtering"],
        "why": "Constraint satisfaction requiring systematic reasoning"
    },

    # ===== REASONING: Good for demonstrating ITS value =====
    {
        "category": "Mathematical Reasoning",
        "difficulty": "Medium",
        "question": "Is 97 a prime number? Explain your reasoning by checking all necessary divisors.",
        "expected_answer": "Yes, 97 is prime. Need to check divisors up to √97 ≈ 9.8. Testing 2,3,5,7 shows none divide 97 evenly.",
        "best_for": ["self_consistency", "beam_search"],
        "why": "Systematic verification with clear answer"
    },
    {
        "category": "Optimization",
        "difficulty": "Hard",
        "question": "A farmer has 100 feet of fence and wants to create a rectangular garden with maximum area. What should the dimensions be? Show your work.",
        "expected_answer": "25 feet × 25 feet (a square). This gives maximum area of 625 square feet.",
        "best_for": ["particle_filtering", "entropic_particle_filtering"],
        "why": "Optimization problem requiring calculus or systematic search"
    },
    {
        "category": "Probability",
        "difficulty": "Medium",
        "question": "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads? Show all possible outcomes.",
        "expected_answer": "3/8 or 0.375. Favorable outcomes: HHT, HTH, THH (3 out of 8 total outcomes)",
        "best_for": ["beam_search", "self_consistency"],
        "why": "Requires enumeration of sample space"
    },
]


def get_questions_by_algorithm(algorithm: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Get example questions best suited for a specific algorithm.

    Args:
        algorithm: Algorithm name (e.g., 'beam_search', 'best_of_n')
        limit: Maximum number of questions to return

    Returns:
        List of question dictionaries
    """
    # Filter questions that are good for this algorithm
    suitable = [q for q in EXAMPLE_QUESTIONS if algorithm in q["best_for"]]

    # If we don't have enough, add others
    if len(suitable) < limit:
        remaining = [q for q in EXAMPLE_QUESTIONS if algorithm not in q["best_for"]]
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
    return [q for q in EXAMPLE_QUESTIONS if q["difficulty"] == difficulty]


def get_all_questions() -> List[Dict[str, str]]:
    """Get all example questions."""
    return EXAMPLE_QUESTIONS
