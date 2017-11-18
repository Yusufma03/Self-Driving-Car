class test:
    class sub:
        def __init__(self):
            print("init sub")

    def __init__(self):
        self.sub = test.sub()
        print("init test")

a = test()
