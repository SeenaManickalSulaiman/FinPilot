import operator
import re
import json
from typing import Annotated, List, Tuple, Union, TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    context: dict


class FinancialAdvisorAgent:
    def __init__(self, tools: List[Tool], api_key: str):
        self.tools = tools
        self.llm = ChatOpenAI(
            api_key=api_key, model="gpt-4o", temperature=0.5
        )
        self.tools_by_name = {tool.name: tool for tool in tools}

        self.system_prompt = (
            "You are a professional financial advisor AI assistant. "
            "You have access to specialized tools like budget planners, "
            "investment analyzers, market trend analysis, and portfolio analyzers. "
            "Use these tools to provide detailed, insightful financial advice with clear, actionable recommendations."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )

    def _extract_tool_usage(self, intermediate_steps):
        tools_used = []
        tool_results = []
        for action, result in intermediate_steps:
            if hasattr(action, "tool"):
                tools_used.append(action.tool)
                tool_results.append(result)
        if tools_used:
            return tools_used[-1], tool_results[-1], tools_used, tool_results
        return None, None, [], []

    def _prepare_tool_input(self, message: str, tool_name: str) -> str:
        """Simplified input preparation for each tool."""
        if tool_name == "investment_analyzer":
            # Extract stock symbols
            symbols = re.findall(r"\b[A-Z]{2,5}\b", message)
            return symbols[0] if symbols else ""
        elif tool_name == "budget_planner":
            # Dummy implementation
            return message
        elif tool_name == "portfolio_analyzer":
            return message
        elif tool_name == "market_trends":
            return message
        return message

    def process_message(self, message: str, history: List[Tuple[str, str]] = None) -> str:
        messages = []
        if history:
            for role, content in history:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        try:
            result = self.agent_executor.invoke({"input": message, "messages": messages})
            output = result.get("output", "I'm sorry, I couldn't process your request.")
            # Enhance the output with actionable insights
            enhanced_response = (
                f"🔎 Here’s my analysis:\n\n"
                f"{output}\n\n"
                "✅ Recommendations:\n"
                "- Review your financial goals regularly.\n"
                "- Diversify investments to manage risk.\n"
                "- Seek professional advice for significant financial decisions."
            )
            return enhanced_response
        except Exception as e:
            return f"⚠️ Error processing your request: {e}"
