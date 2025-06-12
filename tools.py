import json
from datetime import datetime
from typing import List

import yfinance as yf
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults


class FinancialTools:
    def __init__(self, tavily_api_key: str):
        self.tavily_search = TavilySearchResults(api_key=tavily_api_key)

    def create_budget_planner(self) -> Tool:
        def budget_planner(input_str: str) -> str:
            """Create a personalized budget plan with income and expense analysis."""
            try:
                if not input_str or input_str.strip() == "":
                    input_str = '{"income": 5000, "expenses": {}}'

                try:
                    data = json.loads(input_str)
                except json.JSONDecodeError:
                    import re
                    income_match = re.search(r"(\$?[\d,]+(?:\.\d{2})?)", input_str)
                    income = float(income_match.group(1).replace("$", "").replace(",", "")) if income_match else 5000
                    data = {"income": income, "expenses": {}}

                income = data.get("income", 5000)
                expenses = data.get("expenses", {})
                goals = data.get("savings_goals", {})
                debt = data.get("debt", {})

                needs = income * 0.5
                wants = income * 0.3
                savings = income * 0.2
                total_expenses = sum(expenses.values())
                remaining = income - total_expenses
                total_debt = sum(debt.values()) if debt else 0
                debt_to_income = (total_debt / income * 100) if income > 0 else 0
                emergency_fund_needed = total_expenses * 6
                emergency_fund_goal = goals.get("emergency_fund", 0)
                debt_payments = debt.get("monthly_payments", 0)
                available_for_savings = remaining - debt_payments

                budget_plan = {
                    "monthly_income": income,
                    "recommended_allocation": {
                        "needs": needs,
                        "wants": wants,
                        "savings": savings,
                    },
                    "current_expenses": expenses,
                    "total_expenses": total_expenses,
                    "remaining_budget": remaining,
                    "savings_rate": (available_for_savings / income * 100) if income > 0 else 0,
                    "debt_analysis": {
                        "total_debt": total_debt,
                        "debt_to_income_ratio": debt_to_income,
                        "monthly_payments": debt_payments,
                    },
                    "emergency_fund": {
                        "recommended": emergency_fund_needed,
                        "current": emergency_fund_goal,
                        "progress": (emergency_fund_goal / emergency_fund_needed * 100)
                        if emergency_fund_needed > 0 else 0,
                    },
                    "savings_optimization": {
                        "available_monthly": available_for_savings,
                        "annual_savings_potential": available_for_savings * 12,
                    },
                    "recommendations": [],
                }

                if available_for_savings < savings:
                    budget_plan["recommendations"].append(
                        f"Consider increasing savings by approximately ${savings - available_for_savings:.2f} per month to meet your 20% savings target."
                    )
                if debt_to_income > 36:
                    budget_plan["recommendations"].append(
                        f"High debt-to-income ratio ({debt_to_income:.1f}%). Consider reducing debt or restructuring loans."
                    )
                if emergency_fund_goal < emergency_fund_needed:
                    monthly_needed = (emergency_fund_needed - emergency_fund_goal) / 12
                    budget_plan["recommendations"].append(
                        f"Increase your emergency fund by saving ${monthly_needed:.2f} per month for the next 12 months."
                    )
                largest_expense = max(expenses.items(), key=lambda x: x[1]) if expenses else None
                if largest_expense and largest_expense[1] > income * 0.35:
                    budget_plan["recommendations"].append(
                        f"Your {largest_expense[0]} expense (${largest_expense[1]:.2f}) is high. Consider reducing this cost."
                    )

                return json.dumps(budget_plan, indent=2)
            except Exception as e:
                return f"Error creating budget plan: {str(e)}"

        return Tool(
            name="budget_planner",
            description="Create personalized budget plans with income and expense analysis.",
            func=budget_planner,
        )

    def create_investment_analyzer(self) -> Tool:
        def investment_analyzer(symbol: str) -> str:
            """Analyze stocks and provide investment recommendations."""
            try:
                stock = yf.Ticker(symbol.upper())
                info = stock.info
                hist = stock.history(period="1y")

                if hist.empty:
                    return f"No data available for {symbol}."

                current_price = info.get("currentPrice", hist["Close"].iloc[-1])
                pe_ratio = info.get("trailingPE", "N/A")
                dividend_yield = (info.get("dividendYield", 0) * 100) if info.get("dividendYield") else 0
                market_cap = info.get("marketCap", "N/A")
                beta = info.get("beta", "N/A")
                sector = info.get("sector", "Unknown")
                rsi = 100 - (100 / (1 + (hist["Close"].diff().clip(lower=0).rolling(window=14).mean() /
                                        hist["Close"].diff().clip(upper=0).abs().rolling(window=14).mean()))).iloc[-1]

                analysis = {
                    "symbol": symbol.upper(),
                    "company_name": info.get("longName", symbol),
                    "sector": sector,
                    "current_price": f"${current_price:.2f}",
                    "market_cap": f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else "N/A",
                    "pe_ratio": pe_ratio,
                    "dividend_yield": f"{dividend_yield:.2f}%",
                    "beta": beta,
                    "rsi": f"{rsi:.1f}",
                    "recommendation": "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD",
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"Error analyzing stock {symbol}: {str(e)}"

        return Tool(
            name="investment_analyzer",
            description="Analyze stocks with fundamental metrics, risk, and recommendations.",
            func=investment_analyzer,
        )

    def create_market_trends_analyzer(self) -> Tool:
        def market_trends(query: str) -> str:
            """Provide an overview of current market trends and financial news."""
            try:
                current_year = datetime.now().year
                comprehensive_query = f"{query} stock market trends analysis {current_year}"
                market_news = self.tavily_search.run(comprehensive_query)

                analysis = {
                    "query": query,
                    "market_news": market_news,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                return json.dumps(analysis, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"Error analyzing market trends: {str(e)}"

        return Tool(
            name="market_trends",
            description="Get real-time market trends and financial news.",
            func=market_trends,
        )

    def create_portfolio_analyzer(self) -> Tool:
        def portfolio_analyzer(input_str: str) -> str:
            """Analyze portfolio performance and diversification."""
            try:
                data = json.loads(input_str) if input_str.strip().startswith("{") else {}
                holdings = data.get("holdings", [])
                total_investment = data.get("total_investment", 0)

                if not holdings:
                    return "No valid holdings found. Please provide a valid portfolio."

                portfolio_data = []
                for holding in holdings:
                    symbol = holding.get("symbol", "")
                    shares = holding.get("shares", 0)
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1d")
                    current_price = hist["Close"].iloc[-1] if not hist.empty else 0
                    value = shares * current_price
                    portfolio_data.append({
                        "symbol": symbol,
                        "shares": shares,
                        "current_price": f"${current_price:.2f}",
                        "value": value,
                    })

                analysis = {
                    "total_investment": total_investment,
                    "portfolio_value": sum(item["value"] for item in portfolio_data),
                    "holdings": portfolio_data,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"Error analyzing portfolio: {str(e)}"

        return Tool(
            name="portfolio_analyzer",
            description="Analyze portfolio performance and diversification.",
            func=portfolio_analyzer,
        )

    def get_all_tools(self) -> List[Tool]:
        return [
            self.create_budget_planner(),
            self.create_investment_analyzer(),
            self.create_market_trends_analyzer(),
            self.create_portfolio_analyzer(),
        ]
