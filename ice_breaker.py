from typing import Tuple

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup import lookup as linkedin_lookup_agent
from output_parsers import summary_parser, Summary


# Function to create an icebreaker with LinkedIn data
def ice_break_with(name: str) -> Tuple[Summary, str]:
    # Look up the LinkedIn username using the provided name
    linkedin_username = linkedin_lookup_agent(name=name)
    # Scrape the LinkedIn profile data using the username (mock data used for demonstration)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=False
    )

    # Define the summary template
    summary_template = """
    Given the LinkedIn information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    \n{format_instructions}
    """
    # Create a PromptTemplate instance with the defined template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature="0", model="gpt-3.5-turbo")
    # Create a processing chain using the prompt template and the LLM
    chain = summary_prompt_template | llm | summary_parser
    # Invoke the chain with the LinkedIn data
    res: Summary = chain.invoke(input={"information": linkedin_data})

    return res, linkedin_data.get("profile_pic_url")


# Main entry point of the script
if __name__ == "__main__":
    print("Hello LangChain!")
    ice_break_with("Nirmiti karandhikar")
