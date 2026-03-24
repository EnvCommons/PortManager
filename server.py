import sys

from openreward.environments import Server

from portmanager import PortManager

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8080
    server = Server([PortManager])
    server.run(port=port)
