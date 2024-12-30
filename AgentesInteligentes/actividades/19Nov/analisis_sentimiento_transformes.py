from transformers import pipeline

sentimental_analysis = pipeline("sentiment-analysis")

result = sentimental_analysis("Dios mio! esto es muy rapido!")

print(result)