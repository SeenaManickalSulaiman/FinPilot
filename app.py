import os
import gradio as gr
from PIL import Image
from financial_agent import FinancialAdvisorAgent
from tools import FinancialTools

# API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

if not OPEN_API_KEY:
    raise EnvironmentError("🚨 OPEN_API_KEY is not set. Please set it in your environment variables.")

# Initialize tools and agent
financial_tools = FinancialTools(tavily_api_key=TAVILY_API_KEY)
tools = financial_tools.get_all_tools()
agent = FinancialAdvisorAgent(tools=tools, api_key=OPEN_API_KEY)

# Load sidebar image
try:
    sidebar_image = Image.open("sidebar_image1.jpg")
except Exception:
    sidebar_image = None

# Process user queries
def process_financial_query(message, history):
    if not message.strip():
        return history
    response = agent.process_message(message, history)
    history.append(("user", message))
    history.append(("assistant", response))
    return history

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Row():
        # Sidebar (left)
        with gr.Column(scale=1):
            if sidebar_image:
                gr.Image(
                    sidebar_image,
                    elem_id="sidebar-image",
                    show_label=False,
                    container=False,
                    scale=0.8
                )
            gr.Markdown("### 📖 About FinPilot")
            gr.Markdown("""
FinPilot is your AI-powered financial assistant.  
Ask questions about budgeting, investments, market trends, and get real-time insights tailored to your needs.

---

### 🛠️ Instructions:
- **Type your financial question** in the input box.
- **Get smart insights** powered by real-time data, analytics, and financial best practices.
- **Explore example queries** if you're unsure what to ask.
- **Combine budgeting + investment** queries for deeper, more personalized insights.

Developed by [Seena MS](https://www.linkedin.com/in/seenams/)
""")

        # Main Chat Area (right)
        with gr.Column(scale=3):
            gr.Markdown("## 💼 FinPilot — Your Financial AI Co-Pilot")
            gr.Markdown("Ask any financial question and get real-time insights with helpful analysis.")

            chatbot = gr.Chatbot(
                height=300
            )
            msg = gr.Textbox(
                label="📄 Type your question below:",
                placeholder="Ask your financial question here..."
            )
            submit = gr.Button("Submit", variant="primary")

            # Example questions
            gr.Markdown("""
### 💡 Example Questions
- How can I allocate my $5000 monthly income using the 50/30/20 rule?
- Should I invest in Tesla stock right now?
- What are the latest trends in renewable energy stocks?
- How can I diversify my portfolio to reduce risk?
- What's the current market sentiment for tech stocks?
""")

            # Handle submit
            def handle_submission(message, history):
                history = process_financial_query(message, history)
                return "", history

            submit.click(handle_submission, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
