from importlib import import_module

nps=import_module("numpy")
print(nps.random.random())