from dataclasses import dataclass

@dataclass
class ServiceRequest:
    sender: str
    receiver: str
    service_type: str