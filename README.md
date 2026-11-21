# FORGE-200M
### Experimental Language Model Research

**FORGE-200M** (Foundation for Open Research in Generative Engineering) is an experimental language model development project focused on building and studying compact decoder-only Transformer architectures from scratch.

The project currently features a custom 200-million-parameter Transformer model, trained on approximately 1 billion tokens, with ongoing experiments in supervised fine-tuning and structured tool calling.

This repository documents the research, implementation decisions, training experiments, evaluation results, and ongoing development of the model.

> **Project Status: Early Research / Work in Progress**
>
> FORGE-200M is not a production-ready language model. Its capabilities, particularly structured tool calling, are still being investigated and evaluated. The current implementation should be considered experimental.

## Research Objectives

- Develop a decoder-only Transformer architecture from scratch.
- Explore efficient attention mechanisms such as Grouped Query Attention.
- Build and experiment with a custom tokenizer.
- Study language-model pretraining at the 200M parameter scale.
- Investigate supervised fine-tuning for structured function calling.
- Evaluate the feasibility of tool-use capabilities in compact language models.

## Current Milestones

- [x] Custom tokenizer implementation
- [x] Transformer architecture implementation
- [x] 200M parameter model configuration
- [x] Pretraining on approximately 1B tokens
- [x] Initial model evaluation
- [x] Initial tool-calling fine-tuning experiments
- [ ] Extended supervised fine-tuning
- [ ] Structured tool-calling evaluation
- [ ] Multi-tool interaction experiments
- [ ] Generalization and robustness evaluation
- [ ] Experimental model release

**This repository is actively evolving.** Architecture changes, training configurations, experimental results, and implementation details may change as development progresses.
