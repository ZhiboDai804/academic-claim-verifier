from types import SimpleNamespace

import pytest

from src.embeddings import embed_query, embed_texts


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        # Return deliberately out of order to verify index-based ordering.
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
            for index, text in reversed(list(enumerate(input)))
        ]
        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


def test_embed_texts_batches_inputs_and_preserves_order() -> None:
    client = FakeOpenAIClient()

    vectors = embed_texts(["first", "second"], client=client, model="test-model")

    assert vectors == [[0.0, 5.0], [1.0, 6.0]]
    assert client.embeddings.calls == [
        {"model": "test-model", "input": ["first", "second"]}
    ]


def test_embed_query_returns_one_vector() -> None:
    client = FakeOpenAIClient()

    assert embed_query("question", client=client, model="test-model") == [0.0, 8.0]


@pytest.mark.parametrize("texts", [[], [""], ["   "], ["valid", ""]])
def test_embed_texts_rejects_empty_input(texts: list[str]) -> None:
    with pytest.raises(ValueError):
        embed_texts(texts, client=FakeOpenAIClient())


def test_embed_texts_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        embed_texts(["valid"], client=FakeOpenAIClient(), model=" ")
