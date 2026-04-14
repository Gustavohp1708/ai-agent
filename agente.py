import os
from dotenv import load_dotenv
from rich import print
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from indexar import carregar_vectorstore

load_dotenv()

vectorstore = carregar_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.45")),
)

prompt = ChatPromptTemplate.from_template(
    "Você é um assistente interno cordial e claro, que ajuda colaboradores com dúvidas sobre benefícios e políticas.\n\n"
    "Regras sobre o conteúdo:\n"
    "- Use o contexto abaixo como única fonte para fatos (valores, prazos, regras, nomes de planos, procedimentos).\n"
    "- Não copie blocos longos literalmente: explique com suas palavras, de forma organizada (tópicos curtos quando fizer sentido).\n"
    "- Você pode fazer conexões lógicas entre trechos do contexto (ex.: relacionar cobertura e carência), desde que tudo esteja sustentado pelo texto.\n"
    "- Se o contexto for insuficiente ou não responder à pergunta, diga com honestidade que não encontrou isso na base e diga o que *sí* aparece, se houver.\n"
    "- Não invente informações, números ou políticas que não estejam no contexto.\n\n"
    "Tom: educado, profissional e objetivo; evite ser robótico.\n\n"
    "Estrutura da resposta:\n"
    "1) Resposta principal à pergunta.\n"
    "2) Se couber, um parágrafo curto com informações complementares do mesmo tema que estejam no contexto.\n"
    "3) Ao final, obrigatoriamente uma seção assim (em português):\n"    
    "- Se quiser posso te mostrar mais sobre... Sugira de 1 a 2 temas relevantes alinhados ao tema da pergunta atual e ao que o contexto permite inferir.\n"
    "- Se o contexto for muito pobre, sugira perguntas genéricas para aprofundar o mesmo assunto, sem afirmar fatos inexistentes.\n\n"
    "Contexto:\n{context}\n\n"
    "Pergunta do colaborador: {question}"
)

def format_chunks(chunks):
    formatted = []
    for chunk in chunks:
        source = chunk.metadata.get("filename", "arquivo_desconhecido")
        page = chunk.metadata.get("page", "sem_pagina")
        formatted.append(f"[Fonte: {source} | Página: {page}]\n{chunk.page_content}")
    return "\n\n".join(formatted)

def chamar_llm(texto):
    # O modelo e5 funciona melhor com prefixo "query:" na pergunta.
    contexto_docs = retriever.invoke(f"query: {texto}")
    contexto = format_chunks(contexto_docs)
    mensagens = prompt.format_messages(context=contexto, question=texto)
    resultado = llm.invoke(mensagens)
    return resultado.content

if __name__ == "__main__":
    print(chamar_llm("Qual valor do plano de saúde mais basico?"))