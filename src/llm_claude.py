import os
from anthropic import Anthropic

# Initialize Anthropic client using your environment variable
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def call_claude(prompt: str) -> str:
    """
    Sends a prompt to Claude Sonnet 4.6 (the model available to your API key)
    and returns the generated text. Used by the dashboard to produce
    FDA-style narratives.
    """

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Anthropic returns a list of content blocks
        return response.content[0].text

    except Exception as e:
        return f"[Claude API Error] {str(e)}"