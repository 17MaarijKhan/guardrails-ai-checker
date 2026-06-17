from guardrails import Guard
from guardrails.hub import RegexMatch

# Create Guard
guard = Guard().use(
    RegexMatch(regex="^[a-zA-Z0-9 ?.,!]+$", on_fail="exception")
)

# Bad words list
BAD_WORDS = [
    "hell", "damn", "shut up", "hate", "stupid", "idiot",
    "fool", "dumb", "kill", "die", "ugly", "loser", "shut",
    "horrible", "terrible", "awful", "trash", "garbage",
    "worthless", "useless", "moron", "jerk", "freak"
]

def check_input(user_input):
    try:
        # Step 1: Bad words check
        input_lower = user_input.lower()
        for word in BAD_WORDS:
            if word.lower() in input_lower:
                return False, f"❌ BLOCKED - Inappropriate language detected! '{word}' is not allowed!"

        # Step 2: Dangerous code check
        guard.validate(user_input)
        return True, "✅ SAFE - Your input is completely fine!"

    except Exception as e:
        return False, "❌ BLOCKED - Your input contains dangerous code!"

# Interactive Loop
print("=" * 50)
print("   GUARDRAILS AI - Input Safety Checker")
print("=" * 50)
print("This program checks whether your input")
print("is safe or dangerous!")
print()
print("Three types of checks are performed:")
print("  1. Bad words check")
print("  2. SQL Injection check")
print("  3. Script Injection check")
print()
print("Type 'exit' to quit the program")
print("=" * 50)

while True:
    print()
    user_input = input("Enter your input: ")

    if user_input.lower() == "exit":
        print("\nThank you! Program is closing!")
        break

    if user_input.strip() == "":
        print("⚠️  Please enter something!")
        continue

    is_safe, message = check_input(user_input)
    print(message)

    if is_safe:
        print("   This input can be safely sent to AI!")
    else:
        print("   This input was blocked - system is safe!")