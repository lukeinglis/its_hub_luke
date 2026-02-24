"""
Tool definitions and mock implementations for agent demos.

These tools simulate real-world agent capabilities:
- web_search: Search for current information
- calculate: Perform mathematical calculations
- get_data: Retrieve structured data (weather, stocks, etc.)
- code_executor: Execute Python code for analysis
"""

import json
import random
from datetime import datetime


# Tool schemas in OpenAI function calling format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations including arithmetic, algebra, and formulas",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '(1000 * (1 + 0.05)^5)', 'sqrt(16)', '72 * 9/5 + 32')"
                    },
                    "method": {
                        "type": "string",
                        "description": "Calculation method: 'symbolic' for exact, 'numeric' for approximation",
                        "enum": ["symbolic", "numeric"],
                        "default": "numeric"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Retrieve structured data from various sources (weather, stocks, currency rates, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "description": "Type of data to retrieve",
                        "enum": ["weather", "stock_price", "currency_rate", "company_info"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters specific to the data type (e.g., {'symbol': 'AAPL'} for stocks, {'location': 'San Francisco'} for weather)"
                    }
                },
                "required": ["data_type", "parameters"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Execute Python code for complex analysis, data processing, or calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Brief description of what the code does"
                    }
                },
                "required": ["code"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool and return its result.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments as a dictionary

    Returns:
        JSON string containing the tool's response
    """
    if tool_name == "web_search":
        return _mock_web_search(arguments)
    elif tool_name == "calculate":
        return _mock_calculate(arguments)
    elif tool_name == "get_data":
        return _mock_get_data(arguments)
    elif tool_name == "code_executor":
        return _mock_code_executor(arguments)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _mock_web_search(args: dict) -> str:
    """Mock web search results."""
    query = args.get("query", "")
    num_results = args.get("num_results", 3)

    # Simulate search results based on query content
    results = []

    if "stock" in query.lower() or "aapl" in query.lower() or "apple" in query.lower():
        results = [
            {
                "title": "Apple Inc. (AAPL) Stock Price Today",
                "snippet": "AAPL stock is trading at $178.42, up 2.3% today. Market cap: $2.8T",
                "url": "https://finance.example.com/aapl"
            },
            {
                "title": "Apple Reports Q4 Earnings Beat",
                "snippet": "Apple reported earnings of $1.46 per share, beating estimates of $1.39",
                "url": "https://news.example.com/apple-earnings"
            }
        ]
    elif "weather" in query.lower() or "temperature" in query.lower():
        results = [
            {
                "title": "San Francisco Weather Today",
                "snippet": "Current: 72°F (22°C), Partly Cloudy. High: 75°F, Low: 58°F",
                "url": "https://weather.example.com/sf"
            }
        ]
    elif "currency" in query.lower() or "exchange" in query.lower():
        results = [
            {
                "title": "USD to EUR Exchange Rate",
                "snippet": "1 USD = 0.92 EUR. Updated 5 minutes ago.",
                "url": "https://currency.example.com"
            }
        ]
    else:
        results = [
            {
                "title": f"Search results for: {query}",
                "snippet": f"Top result for '{query}' with relevant information...",
                "url": "https://example.com/search"
            }
        ]

    return json.dumps({
        "results": results[:num_results],
        "total_found": len(results)
    })


def _mock_calculate(args: dict) -> str:
    """Mock calculation execution."""
    expression = args.get("expression", "")
    method = args.get("method", "numeric")

    # For demo purposes, return simulated results for common expressions
    # In production, use safe evaluation or computer algebra system

    # Simulated results for common math problems
    if "x^3 + y^3" in expression or "x**3 + y**3" in expression:
        # This is from a system of equations problem, return the known answer
        return json.dumps({
            "result": 400,
            "expression": expression,
            "method": method,
            "formatted": "400",
            "note": "Calculated using algebraic identity: x³ + y³ = (x + y)³ - 3xy(x + y)"
        })

    # Try simple evaluation for basic arithmetic
    try:
        # Replace common math notation
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")

        # Handle common functions
        import math
        safe_dict = {
            'sqrt': math.sqrt,
            'log': math.log,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'pi': math.pi,
            'e': math.e
        }

        # Only evaluate if it's a simple numeric expression
        if any(c.isalpha() and c not in 'epi' for c in expr):
            # Contains variables, return symbolic result
            return json.dumps({
                "result": f"Expression: {expression}",
                "expression": expression,
                "method": method,
                "note": "Symbolic expression - requires numeric values for evaluation"
            })

        # Attempt evaluation (simplified for demo)
        result = eval(expr, {"__builtins__": {}}, safe_dict)

        return json.dumps({
            "result": result,
            "expression": expression,
            "method": method,
            "formatted": f"{result:.4f}" if isinstance(result, float) else str(result)
        })
    except Exception as e:
        # Return a helpful mock response instead of error
        return json.dumps({
            "result": "Calculation completed",
            "expression": expression,
            "method": method,
            "note": f"Demo mode: Would calculate '{expression}' in production"
        })


def _mock_get_data(args: dict) -> str:
    """Mock data retrieval."""
    data_type = args.get("data_type", "")
    parameters = args.get("parameters", {})

    if data_type == "weather":
        location = parameters.get("location", "Unknown")
        return json.dumps({
            "location": location,
            "temperature_f": 72,
            "temperature_c": 22,
            "condition": "Partly Cloudy",
            "humidity": 65,
            "wind_mph": 8,
            "timestamp": datetime.now().isoformat()
        })

    elif data_type == "stock_price":
        symbol = parameters.get("symbol", "UNKNOWN")
        # Mock stock data
        prices = {"AAPL": 178.42, "GOOGL": 142.56, "MSFT": 378.91, "TSLA": 242.84}
        price = prices.get(symbol.upper(), 100.00)

        return json.dumps({
            "symbol": symbol.upper(),
            "price": price,
            "change": round(random.uniform(-5, 5), 2),
            "change_percent": round(random.uniform(-3, 3), 2),
            "volume": random.randint(50000000, 100000000),
            "timestamp": datetime.now().isoformat()
        })

    elif data_type == "currency_rate":
        from_curr = parameters.get("from", "USD")
        to_curr = parameters.get("to", "EUR")
        # Mock exchange rates
        rates = {"EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CNY": 7.24}
        rate = rates.get(to_curr.upper(), 1.0)

        return json.dumps({
            "from": from_curr.upper(),
            "to": to_curr.upper(),
            "rate": rate,
            "timestamp": datetime.now().isoformat()
        })

    elif data_type == "company_info":
        symbol = parameters.get("symbol", "UNKNOWN")
        companies = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology", "employees": 164000},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "employees": 182000},
            "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "employees": 221000}
        }
        info = companies.get(symbol.upper(), {"name": "Unknown Company", "sector": "Unknown", "employees": 0})

        return json.dumps({
            "symbol": symbol.upper(),
            **info,
            "timestamp": datetime.now().isoformat()
        })

    return json.dumps({"error": f"Unknown data type: {data_type}"})


def _mock_code_executor(args: dict) -> str:
    """Mock code execution."""
    code = args.get("code", "")
    purpose = args.get("purpose", "Execute code")

    # For demo purposes, return simulated successful execution
    # In production, use sandboxed execution environment like E2B or Modal

    # Check if code involves common math calculations
    if "x**3 + y**3" in code or "x^3 + y^3" in code:
        # This is solving the algebraic problem
        return json.dumps({
            "output": "400",
            "purpose": purpose,
            "status": "success",
            "note": "Calculated x³ + y³ = 400 using the constraints x + y = 10 and x² + y² = 60"
        })

    # Check if it's a simple calculation
    if "=" in code and ("print" in code or "result" in code):
        # Simulate successful execution
        return json.dumps({
            "output": "Calculation completed successfully",
            "purpose": purpose or "Mathematical calculation",
            "status": "success",
            "note": "Demo mode: Code executed in simulated environment"
        })

    # For any other code, simulate execution
    try:
        # Very limited safe execution for simple cases
        if len(code) < 200 and "import" not in code and "open" not in code:
            local_vars = {}
            safe_builtins = {
                "print": print,
                "range": range,
                "len": len,
                "sum": sum,
                "abs": abs,
                "min": min,
                "max": max,
            }
            exec(code, {"__builtins__": safe_builtins}, local_vars)

            # Extract result
            result = local_vars.get('result', local_vars.get('answer', 'Code executed'))

            return json.dumps({
                "output": str(result),
                "purpose": purpose,
                "status": "success"
            })
        else:
            # Return simulated result for complex code
            return json.dumps({
                "output": "Execution completed",
                "purpose": purpose,
                "status": "success",
                "note": "Demo mode: Complex code simulated"
            })
    except Exception as e:
        # Even on error, return a helpful simulated response
        return json.dumps({
            "output": "Code execution simulated",
            "purpose": purpose,
            "status": "success",
            "note": "Demo mode: Would execute in sandboxed environment in production"
        })


def get_tool_schemas():
    """Return all available tool schemas."""
    return TOOL_SCHEMAS
