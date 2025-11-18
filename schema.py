from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source class for the input"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Agent response class for the output"""

    response: str = Field(description="The response from the agent")
    sources: List[Source] = Field(
        default_factory=list, description="The sources of the response"
    )
