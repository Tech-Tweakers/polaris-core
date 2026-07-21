#!/usr/bin/env python3
"""
Exemplo de uso do polaris_core em um projeto Python
Este arquivo mostra como configurar o ambiente para usar o polaris_core
"""

import os
import sys

# ============================================================
# Configuração do ambiente (copie esta parte para seu projeto)
# ============================================================

# Adicionar diretório atual ao path do Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configurar LD_LIBRARY_PATH para encontrar as bibliotecas compartilhadas
lib_dir = os.path.join(current_dir, "polaris_libs")
if os.path.exists(lib_dir):
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ["LD_LIBRARY_PATH"]
    else:
        os.environ["LD_LIBRARY_PATH"] = lib_dir

# Importar polaris_core
try:
    import polaris_core as pc

    print("✅ polaris_core importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar polaris_core: {e}")
    print("\n💡 Certifique-se de que:")
    print("   1. polaris_core.so está no mesmo diretório deste script")
    print("   2. polaris_libs/ contém todas as bibliotecas .so necessárias")
    print("   3. LD_LIBRARY_PATH está configurado corretamente")
    sys.exit(1)

# ============================================================
# Exemplo de uso
# ============================================================


def exemplo_basico():
    """Exemplo básico de uso do polaris_core"""

    # Caminho para o modelo (ajuste conforme necessário)
    model_path = "../../../polaris-ai-v3/models/Polaris-Nebula-8b.q5_k.gguf"

    if not os.path.exists(model_path):
        print(f"⚠️  Modelo não encontrado: {model_path}")
        print("   Ajuste o caminho do modelo no código")
        return

    print("\n🚀 Inicializando engine...")

    # Criar engine com GPU (padrão: todas as camadas na GPU)
    eng = pc.Engine(
        model_path,
        n_ctx=4096,  # Tamanho do contexto
        n_threads=0,  # 0 = usar padrão do sistema
        n_gpu_layers=-1,  # -1 = usar todas as camadas na GPU (padrão)
    )

    print("✅ Engine inicializado!")

    # Callback para streaming
    def on_chunk(chunk: bytes) -> None:
        """Callback chamado a cada chunk de texto gerado"""
        print(chunk.decode("utf-8", errors="ignore"), end="", flush=True)

    print("\n📝 Gerando texto...\n")

    # Gerar texto
    eng.generate(
        prompt="Conte um causo curtinho de vaqueiro.",
        system_prompt="",  # vazio = sem template de chat
        n_predict=120,  # número máximo de tokens
        temperature=0.8,  # criatividade
        top_p=0.95,  # diversidade
        repeat_penalty=1.1,  # penalidade de repetição
        callback=on_chunk,  # streaming
    )

    print("\n\n✅ Geração concluída!")


def exemplo_sem_streaming():
    """Exemplo sem streaming (retorna tudo de uma vez)"""

    model_path = "../../../polaris-ai-v3/models/Polaris-Nebula-8b.q5_k.gguf"

    if not os.path.exists(model_path):
        print(f"⚠️  Modelo não encontrado: {model_path}")
        return

    print("\n🚀 Inicializando engine...")
    eng = pc.Engine(model_path, n_ctx=4096, n_gpu_layers=-1)

    print("📝 Gerando texto (sem streaming)...\n")

    # Sem callback = retorna tudo de uma vez
    resultado = eng.generate(
        prompt="Explique o que é inteligência artificial em uma frase.",
        n_predict=50,
        temperature=0.7,
    )

    print(resultado if resultado else "")
    print("\n✅ Geração concluída!")


def exemplo_cpu_apenas():
    """Exemplo forçando uso apenas de CPU"""

    model_path = "../../../polaris-ai-v3/models/Polaris-Nebula-8b.q5_k.gguf"

    if not os.path.exists(model_path):
        print(f"⚠️  Modelo não encontrado: {model_path}")
        return

    print("\n🚀 Inicializando engine (CPU apenas)...")

    # n_gpu_layers=0 força uso apenas de CPU
    eng = pc.Engine(model_path, n_ctx=4096, n_gpu_layers=0)  # 0 = CPU apenas

    print("📝 Gerando texto...\n")

    resultado = eng.generate(prompt="O que é Python?", n_predict=50)

    print(resultado if resultado else "")
    print("\n✅ Geração concluída!")


if __name__ == "__main__":
    print("=" * 60)
    print("Exemplo de uso do polaris_core")
    print("=" * 60)

    # Escolha qual exemplo executar
    exemplo_basico()
    # exemplo_sem_streaming()
    # exemplo_cpu_apenas()
