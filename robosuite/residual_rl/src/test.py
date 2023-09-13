import numpy as np

class A:
    def __init__ (self):
        self.var_a=100
        self.var_b=50

    def testadd(self):
        self.var_c=90
        return (print(f"result:{self.var_a+20}"))
    
    def test2(self):
        return print(f"result:{self.var_b+20}")
    

class B(A):
    def __init__(self):
        super().__init__()
        self.var_a=500
        self.var_b=1000
    
    def testadd(self):
        super().testadd()
        return (print(f"result:{self.var_c+50}"))
    
B().test2()