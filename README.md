# Smirk: A Tokenizer for OpenSMILES

This repository is a fork of Smirk. Visitors looking for the original project should see
[BattModels/smirk](https://github.com/BattModels/smirk).

<div align="center" display="flex" >

![GitHub License](https://img.shields.io/github/license/BattModels/smirk)
<a href="https://arxiv.org/abs/2409.15370">![arXiv:2409.15370](https://img.shields.io/badge/cs.LG-2409.15370-b31b1b?style=flat&amp;logo=arxiv&amp;logoColor=red)</a>

</div>

Smirk is a chemistry-specific tokenizer that provides complete coverage of the [OpenSMILES](http://opensmiles.org)
specification, that is built using Rust 🦀 and [HuggingFace's tokenizers](https://huggingface.co/docs/tokenizers) 🤗.
Installation is easy, and Smirk works out-of-the-box with the [HuggingFace](https://huggingface.co/docs) ecosystem.

Check our [documentation](https://eeg.engin.umich.edu/smirk) to see `smirk` in action, or [read the paper](https://arxiv.org/abs/2409.15370) to learn
about tokenization for molecular foundation models.

## Fork Runtime Changes

This fork carries runtime and Rust-side changes that are not part of the original project:

- Seed the default Rust `SmirkTokenizer` vocabulary with `[UNK]`.
- Return `tokenizers::Result` from GPE vocabulary/merge updates.
- Map Rust and tokenizer errors to normal Python exceptions.
- Delegate vocabulary sizing to the underlying tokenizers API.
- Use the `add_special_tokens` name consistently for tokenization.
- Expose merged GPE tokens consistently through vocabulary and token-id lookup.
