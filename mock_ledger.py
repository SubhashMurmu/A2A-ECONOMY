class MockLedger:
    def __init__(self):
        self.accounts = {}

    def create_account(self, agent_name):
        self.accounts[agent_name] = 100

    def transfer(self, sender, receiver, amount):
        if self.accounts.get(sender, 0) >= amount:
            self.accounts[sender] -= amount
            self.accounts[receiver] = self.accounts.get(receiver, 0) + amount
            return True
        return False

ledger_instance = MockLedger()
ledger_instance.create_account("AgentA")
ledger_instance.create_account("AgentB")