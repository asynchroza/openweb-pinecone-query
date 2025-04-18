"""
title: Pinecone Query
author: Mihail Bozhilov
author_url: https://asynchroza.com
git_url: https://github.com/asynchroza/openweb-pinecone-query
description: Queries data in Pinecone by embedding queries using small sentence models
required_open_webui_version: 0.6.0
requirements: pinecone
version: 0.1.0
licence: MIT
"""

from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import pinecone


class Tools:
    def __init__(self):
        self.valves = self.Valves()


    class Valves(BaseModel):
        pinecone_api_key: str = Field("", description="Your Pinecone API")
        pinecone_index: str = Field("", description="Pinecone Index Name")
        sentence_transformer_model: str = Field("all-MiniLM-L6-v2", description="Sentence Tranformer Model")

    def search_pinecone_index(
            self,
            query: str
        ) -> str:
        """
        This function retrieves relevant information from a Pinecone index to assist with code-related questions.

        Whenever the LLM receives a query about code—such as how to write a specific function, what a piece of code does, or any other programming-related inquiry—it should call this function with the user's query.

        The function will search the Pinecone index for code snippets and documentation that can help answer the question, explain code behavior, or provide useful context for developers.

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
            file_name = match['metadata'].get('file') 
            code = match['metadata'].get('code') 
            formatted.append(f"Filename: {file_name} -- Code: {code}")

        return "\n\n---\n\n".join(formatted)

if __name__ == "__main__":
    tools = Tools()
    print(tools.search_pinecone_index("How to add item to python list"))

