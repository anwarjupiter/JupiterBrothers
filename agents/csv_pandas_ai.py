import pandas as pd
from langchain_ibm import ChatWatsonx
from constants import *
from langchain_experimental.agents import create_csv_agent

def run(question):
    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    myagent = create_csv_agent(llm=model,path=["input/civil.csv"],allow_dangerous_code=True)
    response = myagent.invoke(question)
    print(response)
if __name__ == "__main__":
    run(question="How many rows totally i have ?")