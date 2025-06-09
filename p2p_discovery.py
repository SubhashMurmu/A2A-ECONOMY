class P2PNetwork:
    def __init__(self):
        self.registry = {}

    def register(self, agent_name, services):
        self.registry[agent_name] = services

    def discover(self, service_type):
        return [name for name, services in self.registry.items() if service_type in services]