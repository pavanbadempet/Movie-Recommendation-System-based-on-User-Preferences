import { WordPieceTokenizer } from "../tokenizer";

describe("WordPieceTokenizer", () => {
  const dummyVocab = `
[PAD]
[UNK]
[CLS]
[SEP]
the
dark
knight
inception
##ception
  `.trim();

  test("tokenizes standard words into IDs", () => {
    const tokenizer = new WordPieceTokenizer(dummyVocab);
    const result = tokenizer.tokenize("the dark knight");
    
    // CLS = 2, the = 4, dark = 5, knight = 6, SEP = 3
    expect(result.input_ids).toEqual([2, 4, 5, 6, 3]);
    expect(result.attention_mask).toEqual([1, 1, 1, 1, 1]);
    expect(result.token_type_ids).toEqual([0, 0, 0, 0, 0]);
  });

  test("handles subwords using ## prefix", () => {
    const tokenizer = new WordPieceTokenizer(dummyVocab);
    const result = tokenizer.tokenize("inception");
    
    // CLS = 2, inception = 7, SEP = 3
    expect(result.input_ids).toEqual([2, 7, 3]);
  });

  test("handles unknown words with UNK ID", () => {
    const tokenizer = new WordPieceTokenizer(dummyVocab);
    const result = tokenizer.tokenize("interstellar");
    
    // CLS = 2, interstellar (unknown) = 1, SEP = 3
    expect(result.input_ids).toEqual([2, 1, 3]);
  });
});
