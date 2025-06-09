from communication.message_schema import ServiceRequest
from ledger.mock_ledger import MockLedger

class AgentBase:
    def __init__(self, name, services, ledger: MockLedger):
        self.name = name
        self.services = services
        self.ledger = ledger
        self.balance = 100

    def list_services(self):
        return self.services

    def handle_request(self, request: ServiceRequest):
        if request.service_type in self.services:
            price = self.services[request.service_type]
            if self.ledger.transfer(request.sender, self.name, price):
                return f"{self.name} performed {request.service_type} for {request.sender}"
            else:
                return "Payment failed."
        return "Service not offered."