"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/raw_prompts.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith():
    """Faz pull do prompt leonanluppi/bug_to_user_story_v1 do LangSmith Hub."""
    prompt_name = "leonanluppi/bug_to_user_story_v1"

    print(f"Puxando prompt: {prompt_name}")
    prompt = hub.pull(prompt_name)
    print(f"Prompt carregado com sucesso")

    system_prompt = ""
    user_prompt = ""

    if hasattr(prompt, "messages"):
        for msg in prompt.messages:
            cls_name = msg.__class__.__name__
            if "System" in cls_name:
                system_prompt = msg.prompt.template if hasattr(msg, "prompt") else str(msg)
            elif "Human" in cls_name:
                user_prompt = msg.prompt.template if hasattr(msg, "prompt") else str(msg)

    if not system_prompt and hasattr(prompt, "template"):
        system_prompt = prompt.template

    prompt_data = {
        "bug_to_user_story_v1": {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt or str(prompt),
            "user_prompt": user_prompt or "{bug_report}",
            "version": "v1",
            "created_at": "2025-01-15",
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    return prompt_data


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    required_vars = ["LANGSMITH_API_KEY"]
    if not check_env_vars(required_vars):
        return 1

    try:
        prompt_data = pull_prompts_from_langsmith()

        output_path = "prompts/raw_prompts.yml"
        if save_yaml(prompt_data, output_path):
            print(f"\nPrompt salvo em: {output_path}")

            print("\nConteúdo do system_prompt:")
            sp = prompt_data["bug_to_user_story_v1"]["system_prompt"]
            for line in sp.split("\n")[:10]:
                print(f"      {line}")
            if sp.count("\n") > 10:
                print("      ...")

            return 0
        else:
            print("\n   [ERRO] Erro ao salvar prompt")
            return 1
    except Exception as e:
        print(f"\n   [ERRO] Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
