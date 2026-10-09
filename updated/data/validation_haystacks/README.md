# Validation haystacks

These three English-language corpora are reserved for held-out head-masking
validation. Detection and head-score aggregation must not use them.

Each subdirectory contains one plain UTF-8 `corpus.txt` file and one held-out
`needle.json`. The JSON fields are `case_id`, `needle`, `question`, and
`expected_answer`.

Pass the parent directory to masking with
`--validation-root data/validation_haystacks`. A seeded contiguous offset selects
the starting window within each corpus. The same `--context-seed` therefore
reconstructs identical prompts for baseline, top, bottom, and different head
rankings; changing it selects another window.

## Corpora

- `paul-graham/corpus.txt`: the former `data/PaulGrahamEssays/` collection,
  joined in lexicographic filename order. A filename separator is retained
  between essays.
- `eugene-onegin/corpus.txt`: *Eugene Oneguine [Onegin]* by Alexander Pushkin,
  translated into English by Henry Spalding. Project Gutenberg ebook 23997:
  https://www.gutenberg.org/ebooks/23997
- `hero-of-our-time/corpus.txt`: *A Hero of Our Time* by Mikhail Lermontov,
  translated into English by J. H. Wisdom and Marr Murray. Project Gutenberg
  ebook 913: https://www.gutenberg.org/ebooks/913

The Project Gutenberg header and footer were removed from the two book corpora;
the literary text itself was otherwise left unchanged. Both source editions are
identified by Project Gutenberg as public domain in the USA. Users outside the
USA are responsible for checking the applicable local law.

Paul Graham uses the original San Francisco sandwich / Dolores Park needle
and its original question and expected answer. Its case ID is
`paul-graham-san-francisco`. The two literary needles are deliberately fictional
and absent from their source texts. Do not insert needles
into the corpus files permanently. The experiment builder should insert the
selected needle into an in-memory, token-trimmed context at the requested depth.
