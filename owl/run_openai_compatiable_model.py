# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    WebToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType

from utils import OwlRolePlaying, run_society

from camel.logger import set_log_level

set_log_level(level="DEBUG")

load_dotenv()


def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        OwlRolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    base_model_config = {
        "model_platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        "model_type": "llama-3.3-70b-versatile",  # Using Mixtral model from Groq
        "api_key": "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",
        "url": "https://api.groq.com/openai/v1",
        "model_config_dict": {
            "temperature": 0.4,
            "max_tokens": 4096,
        },
    }

    models = {
        "user": ModelFactory.create(**base_model_config),
        "assistant": ModelFactory.create(**base_model_config),
        "web": ModelFactory.create(**base_model_config),
        "planning": ModelFactory.create(**base_model_config),
        "image": ModelFactory.create(**base_model_config),
    }

    # Configure toolkits
    tools = [
        *WebToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # Comment this out if you don't have google search
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    question = "Go to amazon.com and search for the cheapest laptop with a rtx 3050 graphics card and ryzen 5. Provide the product name, its price, and a brief description of the design and what the general consensus the reviews are saying about the laptop. I have a budget of 80000 INR"

    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
