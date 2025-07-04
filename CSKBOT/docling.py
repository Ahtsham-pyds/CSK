
import os
from pathlib import Path
from tempfile import mkdtemp
from warnings import filterwarnings
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore


EMBED_MODEL = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
embed_dim = len(EMBED_MODEL.get_text_embedding("Burger"))#
print(embed_dim)


reader = DoclingReader()
node_parser = MarkdownNodeParser()
MILVUS_URI = str(Path(mkdtemp())/ 'docling_ahtsham.db')
vector_store = MilvusVectorStore(uri=MILVUS_URI,dim=embed_dim,overwrite=True)
index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)




#QUERY =  'How do you ensure software compliance (licensing) in an organization '
QUERY = "What tools and metrics would you use to monitor server hardware performance?"
result = index.as_query_engine(llm=llm).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])






