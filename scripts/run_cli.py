"""Interactive CLI for vish_agent, for manual testing and demos.

Keeps one ConversationSession for the life of the process, so later turns
can refer back to earlier ones (e.g. "what's my location?" then "what's the
weather there?").
"""

from vish_agent.pipeline import VishPipeline
from vish_agent.session import ConversationSession


def main() -> None:
    pipeline = VishPipeline()
    session = ConversationSession()
    print("vish_agent ready. Type a message (Ctrl+C to exit).")
    while True:
        try:
            raw_user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw_user_input:
            continue
        response = pipeline.run(raw_user_input, session=session)
        print(response.text)


if __name__ == "__main__":
    main()
