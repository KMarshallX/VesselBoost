class test_mo():
    """
    test model
    """
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def out(self, x):
        return x + self.a + self.b + self.c