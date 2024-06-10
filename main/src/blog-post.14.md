# Prompt Engineering

## AI Agent
any system that tasks a language model with controlling a looping workflow and takes actions.

## Reasoning and Action (ReAct) style design
ReAct is a general paradigm that combines reasoning and acting with LLMs. ReAct prompts LLMs to generate verbal reasoning traces and actions for a task. This allows the system to perform dynamic reasoning to create, maintain, and adjust plans for acting while also enabling interaction to external environments (e.g., Wikipedia) to incorporate additional information into the reasoning.  
A basic loop with the following steps:
1. reason and plan actions to take
2. take actions using tools (regular software functions)
3. observe the effects of the tools and re-plan or react as appropriate
![alt text](images/blog14_reasoning_and_acting.png)

## Reference
https://www.promptingguide.ai/techniques/pal

https://langchain-ai.github.io/langgraph/concepts/
https://langchain-ai.github.io/langgraph/tutorials/introduction/
