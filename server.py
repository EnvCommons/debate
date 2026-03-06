from openreward.environments import Server
from env import DebateEnvironment

if __name__ == "__main__":
    server = Server([DebateEnvironment])
    server.run()
