import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import corpus_chrf

nltk.download("punkt", quiet=True)  # BLEU 需要分词模块

def compute_bleu(preds, refs):

    refs_tokenized = [[ref.split()] for ref in refs]
    preds_tokenized = [pred.split() for pred in preds]
    score = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=SmoothingFunction().method1)
    return round(score, 4)

def compute_chrf(preds, refs):
    # chrF++：直接使用 sacrebleu 实现，支持中文字符级比较
    return round(corpus_chrf(preds, [refs]).score, 4)

def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)  # 关闭词干，避免英文化处理
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(preds, refs)]

    return round(sum(scores) / len(scores), 4)

def evaluate(preds, refs):
    return {
        "BLEU": compute_bleu(preds, refs),
        "chrF++": compute_chrf(preds, refs),
    }
