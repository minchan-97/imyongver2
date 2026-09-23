"""
embedding.py — 단어 임베딩을 '한 번만' 학습해서 고정(freeze)한다.

왜 고정하나:
  이전에 성취기준 학습이 '퍼진(발산)' 원인은, 임베딩을 매 학습마다
  다시 자동생성했기 때문일 가능성이 크다. 임베딩이 계속 흔들리면
  그 위에 얹은 SOM도 기준점이 계속 바뀌어 수렴하지 못한다.

해결:
  임베딩은 전체 국어 자료(L2 코퍼스)로 딱 한 번 학습 → embeddings/gukeo_emb.pkl 로 저장.
  이후 L1/L2/L3/L4 모든 레이어는 이 고정 임베딩을 '읽기 전용'으로 재사용한다.
  → 모든 레이어가 같은 좌표계를 공유하므로 비교·매핑이 일관된다.

방식:
  가벼운 word2vec 스타일(skip-gram, negative sampling)을 numpy로 구현.
  CPU only, 외부 패키지 없음(numpy만). 결정론적(seed 고정).
"""
from __future__ import annotations
CORE_VERSION = "13.2"
import numpy as np
import pickle
from collections import Counter
import sys, os

sys.path.append(os.path.dirname(__file__))
from korean_tokenizer import tokenize

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



class FrozenEmbedding:
    """학습이 끝나면 vectors 를 읽기 전용으로만 쓴다."""
    def __init__(self, word2idx, vectors):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.vectors = vectors            # (V, dim), L2-normalized
        self.dim = vectors.shape[1]

    def vec(self, word: str):
        i = self.word2idx.get(word)
        if i is None:
            return None
        return self.vectors[i]

    def embed_tokens(self, tokens: list[str]):
        """토큰 리스트 → 각 토큰 벡터들의 평균(문장/문단 벡터). OOV는 건너뜀."""
        vs = [self.vectors[self.word2idx[t]] for t in tokens if t in self.word2idx]
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def most_similar(self, word: str, topn=10):
        v = self.vec(word)
        if v is None:
            return []
        sims = self.vectors @ v
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self.idx2word[i] == word:
                continue
            out.append((self.idx2word[i], float(sims[i])))
            if len(out) >= topn:
                break
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "vectors": self.vectors}, f)
        _cloud_push(path)

    @staticmethod
    def load(path) -> "FrozenEmbedding":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return FrozenEmbedding(d["word2idx"], d["vectors"])


def train_embedding(
    texts: list[str],
    dim: int = 64,
    window: int = 3,
    min_count: int = 2,
    epochs: int = 5,
    neg: int = 5,
    lr: float = 0.025,
    seed: int = 42,
) -> FrozenEmbedding:
    """
    skip-gram + negative sampling. numpy only, 결정론적.
    texts: 국어 자료 문장 리스트(L2 코퍼스 전체를 넣는다).
    """
    rng = np.random.default_rng(seed)

    # 1) 토큰화 + 어휘 구축
    tokenized = [tokenize(t) for t in texts]
    freq = Counter(w for toks in tokenized for w in toks)
    vocab = [w for w, c in freq.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    V = len(word2idx)
    if V < 2:
        raise ValueError(f"어휘가 너무 적음(V={V}). 자료를 더 넣거나 min_count를 낮추세요.")

    # 2) negative sampling 분포 (unigram^0.75)
    freq_arr = np.array([freq[w] for w in sorted(vocab)], dtype=np.float64)
    neg_prob = freq_arr ** 0.75
    neg_prob /= neg_prob.sum()

    # 3) 파라미터 초기화
    W_in = (rng.random((V, dim)).astype(np.float64) - 0.5) / dim
    W_out = np.zeros((V, dim), dtype=np.float64)

    # 4) 학습 페어 생성
    idx_sents = [[word2idx[w] for w in toks if w in word2idx] for toks in tokenized]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    for ep in range(epochs):
        loss_acc = 0.0
        pair_count = 0
        order = rng.permutation(len(idx_sents))
        for si in order:
            sent = idx_sents[si]
            for pos, center in enumerate(sent):
                lo = max(0, pos - window)
                hi = min(len(sent), pos + window + 1)
                for ctx_pos in range(lo, hi):
                    if ctx_pos == pos:
                        continue
                    context = sent[ctx_pos]
                    negs = rng.choice(V, size=neg, p=neg_prob)
                    v_in = W_in[center]
                    # positive
                    score = sigmoid(W_out[context] @ v_in)
                    g_pos = (score - 1.0)
                    # negatives
                    neg_scores = sigmoid(W_out[negs] @ v_in)
                    # gradients
                    grad_in = g_pos * W_out[context] + (neg_scores[:, None] * W_out[negs]).sum(0)
                    W_out[context] -= lr * g_pos * v_in
                    W_out[negs] -= lr * neg_scores[:, None] * v_in[None, :]
                    W_in[center] -= lr * grad_in
                    loss_acc += -np.log(score + 1e-9) - np.log(1 - neg_scores + 1e-9).sum()
                    pair_count += 1
        avg = loss_acc / max(pair_count, 1)
        print(f"  [embed] epoch {ep+1}/{epochs}  avg_loss={avg:.4f}  pairs={pair_count}")

    # 5) L2 정규화 후 고정
    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    vectors = W_in / np.maximum(norms, 1e-9)
    print(f"  [embed] done. V={V}, dim={dim}")
    return FrozenEmbedding(word2idx, vectors)
