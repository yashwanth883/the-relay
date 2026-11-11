from typing import List, Dict, Any

EVAL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q1_attention_pe",
        "category": "attention",
        "question": "What are the exact formulas for the positional encodings in the Transformer, and what range of wavelengths do they form?",
        "expected_paper": "attention",
        "expected_page": 6,
        "is_unanswerable": False,
        "keywords": ["positional", "encoding", "sin", "cos", "10000", "wavelength"]
    },
    {
        "id": "q2_attention_bleu",
        "category": "attention",
        "question": "What was the BLEU score achieved by the big Transformer model on the WMT 2014 English-to-German and English-to-French translation tasks?",
        "expected_paper": "attention",
        "expected_page": 8,
        "is_unanswerable": False,
        "keywords": ["28.4", "41.8", "bleu", "wmt", "english-to-german"]
    },
    {
        "id": "q3_attention_warmup",
        "category": "attention",
        "question": "How many training steps and warmup steps were used during training the Transformer base model with the Adam optimizer?",
        "expected_paper": "attention",
        "expected_page": 7,
        "is_unanswerable": False,
        "keywords": ["warmup_steps", "4000", "100,000", "adam"]
    },
    {
        "id": "q4_attention_hardware",
        "category": "attention",
        "question": "What hardware setup and training duration were required for training the big Transformer model?",
        "expected_paper": "attention",
        "expected_page": 8,
        "is_unanswerable": False,
        "keywords": ["8", "p100", "gpus", "3.5 days"]
    },
    {
        "id": "q5_moshi_mimi_specs",
        "category": "moshi",
        "question": "What is the frame rate, bitrate, and algorithmic latency of the Mimi neural audio codec used in Moshi?",
        "expected_paper": "moshi",
        "expected_page": 3,
        "is_unanswerable": False,
        "keywords": ["12.5 hz", "1.1 kbps", "80ms", "mimi"]
    },
    {
        "id": "q6_moshi_codebooks",
        "category": "moshi",
        "question": "How many codebooks and what quantization scheme does the Mimi audio tokenizer use for modeling acoustic tokens?",
        "expected_paper": "moshi",
        "expected_page": 3,
        "is_unanswerable": False,
        "keywords": ["rvq", "residual vector quantization", "codebooks", "32"]
    },
    {
        "id": "q7_moshi_latency",
        "category": "moshi",
        "question": "What is the theoretical latency and practical end-to-end conversation latency achieved by Moshi?",
        "expected_paper": "moshi",
        "expected_page": 1,
        "is_unanswerable": False,
        "keywords": ["160ms", "200ms", "latency", "theoretical"]
    },
    {
        "id": "q8_moshi_duplex",
        "category": "moshi",
        "question": "How does Moshi model multi-stream duplex dialogue between the user and the system simultaneously?",
        "expected_paper": "moshi",
        "expected_page": 4,
        "is_unanswerable": False,
        "keywords": ["duplex", "multi-stream", "user", "agent", "audio"]
    },
    {
        "id": "q9_dynamic_chunking_concept",
        "category": "dynamic_chunking",
        "question": "What is H-Net and how does its dynamic chunking mechanism differ from traditional subword tokenization like BPE?",
        "expected_paper": "dynamic_chunking",
        "expected_page": 1,
        "is_unanswerable": False,
        "keywords": ["h-net", "byte", "tokenizer-free", "bpe", "dynamic chunking"]
    },
    {
        "id": "q10_dynamic_chunking_dna",
        "category": "dynamic_chunking",
        "question": "By how much does H-Net improve data efficiency over baselines on DNA sequences according to the paper?",
        "expected_paper": "dynamic_chunking",
        "expected_page": 1,
        "is_unanswerable": False,
        "keywords": ["4x", "dna", "data efficiency", "improvement"]
    },
    {
        "id": "q11_dynamic_chunking_multistage",
        "category": "dynamic_chunking",
        "question": "What happens when scaling H-Net to multi-stage hierarchies compared to standard token-based Transformers?",
        "expected_paper": "dynamic_chunking",
        "expected_page": 2,
        "is_unanswerable": False,
        "keywords": ["multi-stage", "twice", "size", "scaling"]
    },
    {
        "id": "q12_dynamic_chunking_causality",
        "category": "dynamic_chunking",
        "question": "How does dynamic chunking in H-Net maintain causal structure during autoregressive generation?",
        "expected_paper": "dynamic_chunking",
        "expected_page": 4,
        "is_unanswerable": False,
        "keywords": ["causal", "autoregressive", "boundary", "masking"]
    },
    {
        "id": "q13_cross_pos_alignment",
        "category": "cross_paper",
        "question": "Compare how Attention Is All You Need and Moshi represent positional or temporal ordering across sequences.",
        "expected_paper": "cross_paper",
        "expected_page": 6,
        "is_unanswerable": False,
        "keywords": ["sinusoidal", "temporal", "frames", "stream"]
    },
    {
        "id": "q14_cross_tokenization",
        "category": "cross_paper",
        "question": "How do Moshi's audio tokenizer and H-Net's byte-level modeling represent continuous or un-tokenized input signals?",
        "expected_paper": "cross_paper",
        "expected_page": 3,
        "is_unanswerable": False,
        "keywords": ["mimi", "byte", "rvq", "tokenizer-free"]
    },
    {
        "id": "q15_refusal_rlhf",
        "category": "unanswerable",
        "question": "What discount factor gamma and reward model architecture were used for RLHF and DPO alignment in Moshi?",
        "expected_paper": None,
        "expected_page": None,
        "is_unanswerable": True,
        "keywords": []
    },
    {
        "id": "q16_refusal_gpt4_moe",
        "category": "unanswerable",
        "question": "What is the mixture-of-experts routing algorithm and active parameter count in GPT-4?",
        "expected_paper": None,
        "expected_page": None,
        "is_unanswerable": True,
        "keywords": []
    },
    {
        "id": "q17_refusal_dynamic_chunking_rl",
        "category": "unanswerable",
        "question": "What reinforcement learning policy gradient method was used to optimize chunk boundaries in H-Net?",
        "expected_paper": None,
        "expected_page": None,
        "is_unanswerable": True,
        "keywords": []
    },
    {
        "id": "q18_refusal_quantum",
        "category": "unanswerable",
        "question": "What quantum annealing temperature and qubit connectivity were benchmarked in the Transformer paper?",
        "expected_paper": None,
        "expected_page": None,
        "is_unanswerable": True,
        "keywords": []
    }
]

