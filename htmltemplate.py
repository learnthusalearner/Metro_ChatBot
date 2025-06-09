css = """
<style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f7f9fb;
        color: #333333;
    }
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        max-width: 75%;
        line-height: 1.4;
    }
    .user {
        background-color: #007AFF;      /* Professional blue */
        color: #ffffff;
        margin-left: auto;
        text-align: right;
    }
    .bot {
        background-color: #e5e5ea;      /* Soft gray */
        color: #1c1c1e;
        margin-right: auto;
        text-align: left;
    }
</style>
"""

user_template = """
<div class='chat-bubble user'>
    <strong>You:</strong><br>{{MSG}}
</div>
"""

bot_template = """
<div class='chat-bubble bot'>
    <strong>Assistant:</strong><br>{{MSG}}
</div>
"""
