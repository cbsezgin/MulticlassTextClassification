def token_index(tokens, vocab, missing="<unk>"):
    idx_token = []
    for text in  tokens:
        idx_text = []
        for token in text:
            if token in vocab:
                idx_text.append(vocab.index(token))
            else:
                idx_text.append(vocab.index(missing))
        idx_token.append(idx_text)
    return idx_token