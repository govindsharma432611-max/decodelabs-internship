# ============================================
# Project 1: Rule-Based AI Chatbot
# DecodeLabs Industrial Training | Batch 2026
# ============================================

# --- KNOWLEDGE BASE (Dictionary) ---
responses = {
    "hello": "Hi there! How can I help you?",
    "hi": "Hey! Welcome to DecodeLabs Bot!",
    "how are you": "I'm just a bot, but I'm running perfectly!",
    "what is ai": "AI is the simulation of human intelligence by machines!",
    "what is python": "Python is the most popular language for AI development!",
    "who made you": "I was built by a DecodeLabs intern — a future AI Engineer!",
    "bye": "Goodbye! Have a great day!",
    "help": "Commands: hello, hi, how are you, what is ai, what is python, who made you, bye, quit"
}

# --- WELCOME MESSAGE ---
print("=" * 45)
print("   Welcome to DecodeLabs AI Chatbot 🤖")
print("   Type 'help' to see all commands")
print("   Type 'quit' to exit")
print("=" * 45)

# --- THE HEARTBEAT: INFINITE LOOP ---
while True:

    # Phase 1: Input & Sanitization
    raw_input = input("\nYou: ")
    clean_input = raw_input.lower().strip()

    # Exit Strategy
    if clean_input == "quit":
        print("Bot: Session ended. See you next time! 👋")
        break

    # Phase 2: Lookup + Fallback
    reply = responses.get(clean_input, "❓ I do not understand. Type 'help' to see commands.")
    
    # Phase 3: Output
    print("Bot:", reply)