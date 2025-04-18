"""
title: Pinecone Query
author: Mihail Bozhilov
author_url: https://asynchroza.com
git_url: https://github.com/asynchroza/openweb-pinecone-query
description: Queries data in Pinecone by embedding queries using small sentence models
required_open_webui_version: 0.6.0
requirements: pinecone, sentence_transformers, pydantic, aiohttp
version: 0.1.0
licence: MIT
"""

from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable, Any
import pinecone
import aiohttp


class Tools:
    def __init__(self):
        self.valves = self.Valves()

    class Valves(BaseModel):
        pinecone_api_key: str = Field("", description="Your Pinecone API")
        pinecone_index: str = Field("", description="Pinecone Index Name")
        sentence_transformer_model: str = Field(
            "all-MiniLM-L6-v2", description="Sentence Tranformer Model"
        )

    async def search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search and find the answer to any question in the knowledgebase.

        :param query: The user's code-related question or search string.
        """

        pc = pinecone.Pinecone(api_key=self.valves.pinecone_api_key)
        model = SentenceTransformer(self.valves.sentence_transformer_model)
        index = pc.Index(self.valves.pinecone_index)

        query_embedding = model.encode(query).tolist()
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        if not results.get("matches"):
            return "No results were found"

        formatted = []
        for match in results["matches"]:
            score = match.get("score")
            file_name = match["metadata"].get("file")
            code = match["metadata"].get("code")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Found a match '{file_name}' with score of {score}\n"
                        },
                    }
                )

            formatted.append(f"Filename: {file_name} -- Code: {code}")

        return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    tools = Tools()
    print(tools.search("How to add item to python list"))

