import google.generativeai as genai
import os

# Doğrudan elle ver
genai.configure(api_key="AIzaSyBBTsWbD2dY44iobThx6hyqajIfRdVK2H8")

models = genai.list_models()

for model in models:
    print(model.name, "->", model.supported_generation_methods)
