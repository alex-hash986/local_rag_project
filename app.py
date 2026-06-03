import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

st.set_page_config(page_title="Локальная база знаний (RAG)", layout="wide")


@st.cache_resource
def init_store():
    return InMemoryDocumentStore()


@st.cache_resource
def init_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


document_store = init_store()
doc_converter = init_converter()


@st.cache_resource
def init_pipelines(_store):
    splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=20)
    embedder = OpenAIDocumentEmbedder()

    retrieval = Pipeline()
    retrieval.add_component("text_embedder", OpenAITextEmbedder())
    retrieval.add_component("retriever", InMemoryEmbeddingRetriever(_store))
    retrieval.connect("text_embedder.embedding", "retriever.query_embedding")

    prompt_template = """
    Ответь на вопрос на основе предоставленного контекста.
    Если в контексте нет ответа, скажи, что не знаешь.

    Контекст:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}

    Вопрос: {{ question }}

    Ответ:
    """

    generation = Pipeline()
    generation.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    generation.add_component("llm", OpenAIGenerator())
    generation.connect("prompt_builder", "llm")

    return splitter, embedder, retrieval, generation


splitter, doc_embedder, retrieval_pipeline, generation_pipeline = init_pipelines(
    document_store
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False

with st.sidebar:
    st.title("Загрузка PDF")
    uploaded_files = st.file_uploader(
        "Выберите PDF-файлы", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Обработать документы", type="primary"):
        with st.spinner("Обработка документов..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = doc_converter.convert(tmp_path)
                text = result.document.export_to_text()
                doc = Document(content=text, meta={"source": uploaded_file.name})
                all_docs.append(doc)
                os.unlink(tmp_path)

            split_result = splitter.run(documents=all_docs)
            embed_result = doc_embedder.run(documents=split_result["documents"])
            document_store.write_documents(embed_result["documents"])
            st.session_state.docs_indexed = True
            st.success(f"Обработано {len(uploaded_files)} файлов")

    st.divider()
    st.markdown("**Статус:**")
    if st.session_state.docs_indexed:
        st.info(f"Документы загружены")
    else:
        st.warning("Документы не загружены")

st.title("База знаний (RAG)")
st.markdown("Задайте вопрос по загруженным документам")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.docs_indexed:
        response_text = "Сначала загрузите PDF-документы через боковую панель."
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                retrieval_result = retrieval_pipeline.run(
                    {"text_embedder": {"text": prompt}}
                )
                retrieved_docs = retrieval_result["retriever"]["documents"]

                if not retrieved_docs:
                    response_text = "Не удалось найти информацию по вашему вопросу."
                else:
                    generation_result = generation_pipeline.run(
                        {
                            "prompt_builder": {
                                "documents": retrieved_docs,
                                "question": prompt,
                            }
                        }
                    )
                    response_text = generation_result["llm"]["replies"][0]

                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
