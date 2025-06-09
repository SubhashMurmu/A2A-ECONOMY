class MockIPFS:
    def __init__(self):
        self.storage = {}

    def add(self, content):
        cid = f"CID{len(self.storage)}"
        self.storage[cid] = content
        return cid

    def get(self, cid):
        return self.storage.get(cid, None)
    