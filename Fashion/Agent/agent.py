from uagents import Agent
from chat_proto import chat_proto

agent = Agent()

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()