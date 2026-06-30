export class WordPieceTokenizer {
  private vocab: Record<string, number>;
  private clsTokenId = 101;
  private sepTokenId = 102;
  private unkTokenId = 100;

  constructor(vocabText: string) {
    this.vocab = {};
    const lines = vocabText.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i].trim();
      if (token) {
        this.vocab[token] = i;
      }
    }
    this.clsTokenId = this.vocab["[CLS]"] !== undefined ? this.vocab["[CLS]"] : 101;
    this.sepTokenId = this.vocab["[SEP]"] !== undefined ? this.vocab["[SEP]"] : 102;
    this.unkTokenId = this.vocab["[UNK]"] !== undefined ? this.vocab["[UNK]"] : 100;
  }

  /**
   * Tokenizes text and returns Bert tokenizer inputs.
   */
  tokenize(text: string): { input_ids: number[]; attention_mask: number[]; token_type_ids: number[] } {
    // Lowercase and strip accents
    const cleanText = text
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");

    // Split on whitespace and punctuation
    const words = cleanText.match(/\w+|[^\w\s]/g) || [];
    const tokenIds: number[] = [this.clsTokenId];

    for (const word of words) {
      let start = 0;
      const end = word.length;
      while (start < end) {
        let curSubstr = "";
        let curId = -1;
        for (let i = end; i > start; i--) {
          let substr = word.substring(start, i);
          if (start > 0) {
            substr = "##" + substr;
          }
          if (this.vocab[substr] !== undefined) {
            curSubstr = substr;
            curId = this.vocab[substr];
            break;
          }
        }

        if (curId === -1) {
          tokenIds.push(this.unkTokenId);
          break;
        } else {
          tokenIds.push(curId);
          start += curSubstr.startsWith("##") ? curSubstr.length - 2 : curSubstr.length;
        }
      }
    }

    tokenIds.push(this.sepTokenId);

    // Limit sequence length to 128 tokens
    const finalIds = tokenIds.slice(0, 128);
    const attentionMask = new Array(finalIds.length).fill(1);
    const tokenTypeIds = new Array(finalIds.length).fill(0);

    return {
      input_ids: finalIds,
      attention_mask: attentionMask,
      token_type_ids: tokenTypeIds,
    };
  }
}
