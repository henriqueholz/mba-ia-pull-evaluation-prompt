"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def prompt_data():
    """Fixture que carrega o prompt v2 para todos os testes."""
    data = load_prompts(str(PROMPT_PATH))
    return data["bug_to_user_story_v2"]


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' não encontrado no YAML"
        system_prompt = prompt_data["system_prompt"].strip()
        assert len(system_prompt) > 0, "system_prompt está vazio"

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona (ex: 'Você é um Product Manager')."""
        system_prompt = prompt_data["system_prompt"].lower()
        role_keywords = [
            "você é um",
            "você é uma",
            "atue como",
            "seu papel",
            "product manager",
            "# papel",
        ]
        has_role = any(kw in system_prompt for kw in role_keywords)
        assert has_role, (
            "O prompt não define uma persona/role. "
            "Inclua algo como 'Você é um Product Manager...' no system_prompt."
        )

    def test_prompt_mentions_format(self, prompt_data):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt_data["system_prompt"].lower()
        format_keywords = [
            "como um",
            "eu quero",
            "para que",
            "user story",
            "critérios de aceitação",
            "dado que",
            "quando",
            "então",
            "given",
            "when",
            "then",
            "markdown",
            "formato",
        ]
        matches = [kw for kw in format_keywords if kw in system_prompt]
        assert len(matches) >= 3, (
            f"O prompt menciona apenas {len(matches)} termos de formato. "
            "Inclua instruções claras sobre o formato User Story e Given-When-Then."
        )

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt_data["system_prompt"].lower()
        example_keywords = [
            "exemplo",
            "entrada",
            "saída",
            "example",
            "input",
            "output",
        ]
        has_examples = sum(1 for kw in example_keywords if kw in system_prompt) >= 2
        assert has_examples, (
            "O prompt não contém exemplos de entrada/saída (few-shot). "
            "Adicione pelo menos 2 exemplos práticos."
        )

    def test_prompt_no_todos(self, prompt_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        system_prompt = prompt_data.get("system_prompt", "")
        user_prompt = prompt_data.get("user_prompt", "")
        description = prompt_data.get("description", "")

        full_text = f"{system_prompt} {user_prompt} {description}"
        assert "[TODO]" not in full_text, "Encontrado [TODO] no prompt. Remova antes de submeter."

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied", [])
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}. "
            "Adicione o campo 'techniques_applied' com pelo menos 2 técnicas."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
