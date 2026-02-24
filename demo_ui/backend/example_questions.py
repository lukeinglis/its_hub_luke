"""
Example questions for the ITS demo.

Hand-curated and live-tested against gpt-3.5-turbo with self_consistency,
best_of_n, and beam_search. Each question's best_for list is ordered by
which algorithm most clearly demonstrates improvement.

The get_questions_by_algorithm() function returns questions sorted so the
ones that best showcase the selected algorithm appear first.

All expected answers verified computationally.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


CURATED_QUESTIONS: List[Dict] = [
    # =========================================================================
    # QUESTIONS WITH VERIFIED ITS IMPROVEMENT
    # These are the key demo questions — ITS clearly outperforms baseline.
    # =========================================================================
    {
        "category": "Probability",
        "difficulty": "Easy",
        "question": (
            "Alice and Bob each independently roll a standard six-sided die. "
            "What is the probability that the product of their rolls is even? "
            "Express your answer as a common fraction."
        ),
        "expected_answer": "\\frac{3}{4}",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Baseline often answers 1/2 (wrong). Self-consistency corrects to 3/4 via majority vote.",
        "source": "curated",
        "source_id": "curated-prob-2",
    },
    {
        "category": "Competition Math",
        "difficulty": "Hard",
        "question": (
            "Let $x$ and $y$ be positive real numbers such that "
            "$x + y = 10$ and $x^2 + y^2 = 60$. Find the value of $x^3 + y^3$."
        ),
        "expected_answer": "400",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Baseline often gets wrong answer (100 or other). Both self-consistency and best-of-n reliably find 400.",
        "source": "curated",
        "source_id": "curated-comp-1",
    },
    {
        "category": "Sequences",
        "difficulty": "Medium",
        "question": (
            "In an arithmetic sequence, the 5th term is 23 and the 12th "
            "term is 58. What is the 20th term?"
        ),
        "expected_answer": "98",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Baseline gets wrong answer ~33% of the time. Self-consistency majority vote boosts accuracy to ~83%.",
        "source": "curated",
        "source_id": "curated-seq-1",
    },

    # =========================================================================
    # RELIABLE QUESTIONS — both baseline and ITS solve correctly.
    # Good for showing the system works and for live demos where you
    # need a question that won't fail unexpectedly.
    # =========================================================================
    {
        "category": "Probability",
        "difficulty": "Easy",
        "question": (
            "A box contains 3 red balls, 4 blue balls, and 5 green balls. "
            "Two balls are drawn at random without replacement. "
            "What is the probability that both balls are the same color? "
            "Express your answer as a common fraction."
        ),
        "expected_answer": "\\frac{19}{66}",
        "best_for": ["self_consistency", "best_of_n", "beam_search"],
        "why": "Counting problem — models sometimes miscount combinations",
        "source": "curated",
        "source_id": "curated-prob-1",
    },
    {
        "category": "Algebra",
        "difficulty": "Easy",
        "question": (
            "If $f(x) = 2x + 3$ and $g(x) = x^2 - 1$, "
            "what is the value of $f(g(f(2)))$?"
        ),
        "expected_answer": "99",
        "best_for": ["self_consistency", "best_of_n", "beam_search"],
        "why": "Nested function evaluation — straightforward but tests multi-step computation",
        "source": "curated",
        "source_id": "curated-alg-1",
    },
    {
        "category": "Number Theory",
        "difficulty": "Medium",
        "question": (
            "Find the largest prime factor of $3^8 - 1$."
        ),
        "expected_answer": "41",
        "best_for": ["self_consistency", "best_of_n", "beam_search", "particle_filtering"],
        "why": "Requires difference-of-squares factoring — all algorithms handle this well",
        "source": "curated",
        "source_id": "curated-nt-2",
    },
    {
        "category": "Geometry",
        "difficulty": "Medium",
        "question": (
            "A right triangle has legs of length $a$ and $b$ and hypotenuse "
            "of length $c$. If the area of the triangle is 60 and the "
            "perimeter is 40, what is the length of the hypotenuse?"
        ),
        "expected_answer": "17",
        "best_for": ["self_consistency"],
        "why": "System of equations with geometric constraints — self-consistency handles this reliably",
        "source": "curated",
        "source_id": "curated-geom-1",
    },
    {
        "category": "Rates",
        "difficulty": "Medium",
        "question": (
            "Pipe A can fill a tank in 6 hours and Pipe B can fill it in "
            "4 hours. Pipe C can drain the full tank in 12 hours. If all "
            "three pipes are opened simultaneously, how many hours will it "
            "take to fill the empty tank?"
        ),
        "expected_answer": "3",
        "best_for": ["self_consistency", "best_of_n", "beam_search"],
        "why": "Work-rate problem with fractions — tests careful arithmetic",
        "source": "curated",
        "source_id": "curated-rate-1",
    },
    {
        "category": "Combinatorics",
        "difficulty": "Hard",
        "question": (
            "In how many ways can 8 people be seated around a circular "
            "table if 3 specific people (Alice, Bob, and Carol) must all "
            "sit next to each other?"
        ),
        "expected_answer": "720",
        "best_for": ["self_consistency", "best_of_n", "beam_search", "particle_filtering", "entropic_particle_filtering", "particle_gibbs"],
        "why": "Circular permutation with adjacency constraints — requires grouping and careful counting",
        "source": "curated",
        "source_id": "curated-comb-2",
    },
    {
        "category": "Sequences & Series",
        "difficulty": "Hard",
        "question": (
            "Let $a_1 = 1$, $a_2 = 1$, and $a_n = a_{n-1} + a_{n-2}$ for "
            "$n \\geq 3$ (the Fibonacci sequence). Find the remainder when "
            "$a_1^2 + a_2^2 + a_3^2 + \\cdots + a_{10}^2$ is divided by 10."
        ),
        "expected_answer": "5",
        "best_for": ["self_consistency", "best_of_n", "beam_search", "particle_filtering"],
        "why": "Multi-step computation where arithmetic errors compound across Fibonacci terms",
        "source": "curated",
        "source_id": "curated-seq-2",
    },
]


# Tool calling example questions - demonstrate agent tool selection consensus
TOOL_CALLING_QUESTIONS: List[Dict] = [
    # =========================================================================
    # TOOL SELECTION QUESTIONS
    # These demonstrate the value of consensus on tool choice and parameters
    # =========================================================================
    {
        "category": "Finance",
        "difficulty": "Easy",
        "question": (
            "What's the compound annual growth rate (CAGR) if I invest $1000 "
            "and it grows to $2000 in 5 years?"
        ),
        "expected_answer": "14.87%",
        "best_for": ["self_consistency"],
        "why": "Could use 'calculate' OR 'code_executor'. Tool voting shows consensus on best approach.",
        "source": "tool_calling",
        "source_id": "tc-finance-1",
        "expected_tools": ["calculate", "code_executor"],
    },
    {
        "category": "Data Retrieval",
        "difficulty": "Easy",
        "question": (
            "What's the current temperature in San Francisco in Celsius?"
        ),
        "expected_answer": "22°C (approximately)",
        "best_for": ["self_consistency"],
        "why": "Could use 'get_data' (weather API) OR 'calculate' (if temp in F is known). Shows tool selection consensus.",
        "source": "tool_calling",
        "source_id": "tc-data-1",
        "expected_tools": ["get_data", "calculate"],
    },
    {
        "category": "Finance",
        "difficulty": "Medium",
        "question": (
            "What's the current price of Apple (AAPL) stock and how has it "
            "changed in the last quarter?"
        ),
        "expected_answer": "Current price with percentage change",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Requires both 'get_data' for current price and 'web_search' for historical context. Tool parameter voting important.",
        "source": "tool_calling",
        "source_id": "tc-finance-2",
        "expected_tools": ["get_data", "web_search"],
    },
    {
        "category": "Unit Conversion",
        "difficulty": "Easy",
        "question": (
            "If it's currently 72°F outside, what's that in Celsius?"
        ),
        "expected_answer": "22.2°C",
        "best_for": ["self_consistency"],
        "why": "Simple calculation. Could use 'calculate' with formula OR 'get_data' for conversion. Shows tool choice reliability.",
        "source": "tool_calling",
        "source_id": "tc-convert-1",
        "expected_tools": ["calculate", "get_data"],
    },
    {
        "category": "Data Analysis",
        "difficulty": "Medium",
        "question": (
            "Calculate the monthly payment on a $300,000 mortgage at 6.5% "
            "interest over 30 years."
        ),
        "expected_answer": "$1,896 per month",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Complex formula - could use 'calculate' (with proper formula) OR 'code_executor' for clarity. Tool voting reduces errors.",
        "source": "tool_calling",
        "source_id": "tc-finance-3",
        "expected_tools": ["calculate", "code_executor"],
    },
    {
        "category": "Information Retrieval",
        "difficulty": "Medium",
        "question": (
            "What's the exchange rate from USD to EUR today?"
        ),
        "expected_answer": "~0.92 EUR per USD",
        "best_for": ["self_consistency"],
        "why": "Clearly needs 'get_data' for currency rates, but models might try 'web_search'. Tool consensus ensures correct approach.",
        "source": "tool_calling",
        "source_id": "tc-data-2",
        "expected_tools": ["get_data", "web_search"],
    },
    {
        "category": "Computation",
        "difficulty": "Hard",
        "question": (
            "Calculate the standard deviation of the following dataset: "
            "[12, 15, 18, 22, 25, 28, 30, 35, 40, 45]. Show your work."
        ),
        "expected_answer": "~10.96",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Multi-step calculation. 'code_executor' is cleanest but 'calculate' could work. Parameter consensus shows best approach.",
        "source": "tool_calling",
        "source_id": "tc-stats-1",
        "expected_tools": ["code_executor", "calculate"],
    },
    {
        "category": "Mixed",
        "difficulty": "Hard",
        "question": (
            "I bought Microsoft stock at $250 per share. It's now at $378.91. "
            "If I invested $10,000, what's my return percentage and dollar gain?"
        ),
        "expected_answer": "51.56% return, $5,156.40 gain",
        "best_for": ["self_consistency", "best_of_n"],
        "why": "Requires 'get_data' for current price verification AND 'calculate' for returns. Tool sequencing and parameters matter.",
        "source": "tool_calling",
        "source_id": "tc-mixed-1",
        "expected_tools": ["calculate", "get_data"],
    },
]


def get_all_questions() -> List[Dict[str, str]]:
    """
    Get all curated example questions.

    Returns questions with ITS-improvement questions first, then reliable ones.
    """
    return CURATED_QUESTIONS


def get_questions_by_algorithm(algorithm: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Get example questions ordered by how well they showcase the given algorithm.

    Questions where this algorithm is listed first in best_for appear at the top
    (these are the ones tested to show the clearest improvement). Questions where
    the algorithm appears later are listed next, followed by any remaining.

    Args:
        algorithm: Algorithm name (e.g., 'beam_search', 'best_of_n')
        limit: Maximum number of questions to return

    Returns:
        List of question dictionaries, ordered by demo effectiveness.
    """
    all_questions = get_all_questions()

    # Tier 1: Algorithm is the FIRST entry in best_for (best demo for this algo)
    tier1 = [q for q in all_questions
             if q["best_for"] and q["best_for"][0] == algorithm]

    # Tier 2: Algorithm appears in best_for but not first
    tier2 = [q for q in all_questions
             if algorithm in q["best_for"] and q not in tier1]

    # Tier 3: Algorithm not listed, but still available
    tier3 = [q for q in all_questions
             if algorithm not in q["best_for"]]

    result = tier1 + tier2 + tier3
    return result[:limit]


def get_questions_by_difficulty(difficulty: str) -> List[Dict[str, str]]:
    """
    Get example questions by difficulty level.

    Args:
        difficulty: 'Easy', 'Medium', or 'Hard'

    Returns:
        List of question dictionaries
    """
    return [q for q in get_all_questions() if q["difficulty"] == difficulty]


def get_tool_calling_questions() -> List[Dict[str, str]]:
    """
    Get all tool calling example questions.

    These questions demonstrate agent tool selection consensus scenarios.

    Returns:
        List of tool calling question dictionaries
    """
    return TOOL_CALLING_QUESTIONS


def get_tool_calling_questions_by_algorithm(algorithm: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Get tool calling questions ordered by how well they showcase the given algorithm.

    Args:
        algorithm: Algorithm name (e.g., 'self_consistency', 'best_of_n')
        limit: Maximum number of questions to return

    Returns:
        List of question dictionaries, ordered by demo effectiveness.
    """
    all_questions = get_tool_calling_questions()

    # Tier 1: Algorithm is the FIRST entry in best_for
    tier1 = [q for q in all_questions
             if q["best_for"] and q["best_for"][0] == algorithm]

    # Tier 2: Algorithm appears in best_for but not first
    tier2 = [q for q in all_questions
             if algorithm in q["best_for"] and q not in tier1]

    # Tier 3: Algorithm not listed, but still available
    tier3 = [q for q in all_questions
             if algorithm not in q["best_for"]]

    result = tier1 + tier2 + tier3
    return result[:limit]
