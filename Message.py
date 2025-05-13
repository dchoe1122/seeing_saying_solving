class Message:
    def __init__(self, type, content):
        self.content = content
        self.type = type

    def __str__(self):
        return self.content