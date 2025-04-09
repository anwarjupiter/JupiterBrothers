from pandasai import SmartDataframe
from langchain_ibm import ChatWatsonx
from constants import *
import pandas as pd

def run(question):
    df = pd.read_csv("input/civil.csv")
    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    pandas_agent = SmartDataframe(df,config={"llm":model})
    response = pandas_agent.chat(query=question)
    return response

if __name__ == "__main__":
    bot = run(question="How many AC exists in the Ground Floor ?")
    print(bot)
