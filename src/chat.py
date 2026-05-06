import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta. (Pressione Enter vazio ou digite 'sair' para encerrar.)\n")
    while True:
        try:
            q = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"sair", "quit", "exit"}:
            break
        try:
            resp = chain(q)
        except Exception as e:
            print(f"RESPOSTA: [erro] {e}\n")
            continue
        print(f"RESPOSTA: {resp}\n")


if __name__ == "__main__":
    main()
