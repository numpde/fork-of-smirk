use std::collections::{HashMap, HashSet};

use crate::gpe::{GpeTrainer, GPE};
use crate::pre_tokenizers::{split_structure, SmirkPreTokenizer};
use crate::pyerr::{io_err, map_tok_err, runtime_err, type_err, value_err};
use crate::wrapper::{ModelWrapper, PreTokenizerWrapper, TrainerWrapper};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyString};

use regex::Regex;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::processors::template::{Template, TemplateProcessing};
use tokenizers::{self, normalizers, DecoderWrapper, NormalizerWrapper};
use tokenizers::{
    AddedToken, EncodeInput, OffsetReferential, OffsetType, PaddingDirection, PaddingParams,
    PaddingStrategy, PostProcessorWrapper, PreTokenizedString, PreTokenizer, TokenizerBuilder,
    TokenizerImpl, TruncationDirection, TruncationParams, TruncationStrategy,
};

use ahash::AHashMap;

type Tokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

#[pyclass(dict, module = "smirk.smirk", name = "SmirkTokenizer")]
#[derive(Clone)]
pub struct SmirkTokenizer {
    tokenizer: Tokenizer,
}

impl SmirkTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

fn normalizer() -> PyResult<normalizers::Sequence> {
    let steps: Vec<normalizers::NormalizerWrapper> = [
        normalizers::Replace::new("++", "+2").map_err(|e| runtime_err("normalizer", e))?.into(),
        normalizers::Replace::new("--", "-2").map_err(|e| runtime_err("normalizer", e))?.into(),
        normalizers::Strip::new(true, true).into(),
    ]
    .to_vec();
    Ok(normalizers::Sequence::new(steps))
}

#[pymethods]
impl SmirkTokenizer {
    #[new]
    fn __new__() -> PyResult<Self> {
        let mut vocab: AHashMap<String, u32> = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);

        let tokenizer: Tokenizer = TokenizerBuilder::new()
            .with_model(
                WordLevel::builder()
                    .vocab(vocab)
                    .unk_token("[UNK]".to_string())
                    .build()
                    .map_err(|e| runtime_err("__new__", e))?
                    .into(),
            )
            .with_pre_tokenizer(Some(SmirkPreTokenizer::default().into()))
            .with_normalizer(Some(normalizer()?.into()))
            .with_decoder(Some(Fuse::default().into()))
            .build()
            .map_err(|e| runtime_err("__new__", e))?;

        Ok(Self { tokenizer })
    }

    fn __getstate__(&self) -> PyResult<String> {
        serde_json::to_string(&self.tokenizer)
            .map_err(|e| value_err("__getstate__", e))
    }

    fn __setstate__(&mut self, state: &str) -> PyResult<()> {
        self.tokenizer = serde_json::from_str(state).map_err(|e| value_err("__setstate__", e))?;
        Ok(())
    }

    #[staticmethod]
    fn from_vocab(file: &str) -> PyResult<Self> {
        let model =
            WordLevel::from_file(file, "[UNK]".to_string()).map_err(|e| map_tok_err("from_vocab", e))?;
        let tokenizer = TokenizerBuilder::new()
            .with_model(model.into())
            .with_pre_tokenizer(Some(SmirkPreTokenizer::default().into()))
            .with_normalizer(Some(normalizer()?.into()))
            .with_decoder(Some(Fuse::new().into()))
            .build()
            .map_err(|e| runtime_err("from_vocab", e))?;
        Ok(SmirkTokenizer::new(tokenizer))
    }

    fn pretokenize(&self, smile: String) -> PyResult<Vec<String>> {
        let mut pretokenized = PreTokenizedString::from(smile);
        let pretok = self
            .tokenizer
            .get_pre_tokenizer()
            .ok_or_else(|| runtime_err("pretokenize", "Tokenizer has no pre_tokenizer"))?;
        pretok
            .pre_tokenize(&mut pretokenized)
            .map_err(|e| runtime_err("pretokenize", e))?;
        let splits = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect::<Vec<String>>();
        Ok(splits)
    }

    #[pyo3(signature = (smile, add_special_tokens = true))]
    fn encode(&self, smile: String, add_special_tokens: bool) -> PyResult<Encoding> {
        let input = EncodeInput::from(smile);
        let encoding = self
            .tokenizer
            .encode_char_offsets(input, add_special_tokens)
            .map_err(|e| map_tok_err("encode", e))?;
        Ok(Encoding::from(encoding))
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(|e| map_tok_err("decode", e))
    }

    #[pyo3(signature = (examples, add_special_tokens = true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        examples: Vec<Bound<'_, PyString>>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Encoding>> {
        let inputs: Vec<EncodeInput> = examples
            .into_iter()
            .map(|x| EncodeInput::from(x.to_string()))
            .collect();
        // Release the GIL while tokenizing batch
        let out = py.allow_threads(|| {
            self.tokenizer
                .encode_batch_char_offsets(inputs, add_special_tokens)
                .map_err(|e| map_tok_err("encode_batch", e))
        });
        out.map(|encodings| encodings.into_iter().map(Encoding::from).collect())
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        ids: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let sequences = ids.iter().map(|x| &x[..]).collect::<Vec<&[u32]>>();
            Ok(self
                .tokenizer
                .decode_batch(&sequences, skip_special_tokens)
                .map_err(|e| map_tok_err("decode_batch", e))?)
        })
    }

    #[getter]
    fn get_post_processor(&self) -> PyResult<String> {
        if let Some(post_processor) = self.tokenizer.get_post_processor() {
            serde_json::to_string(&post_processor).map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Ok("{}".to_string())
        }
    }

    #[setter]
    fn set_post_processor(&mut self, template: String) -> PyResult<()> {
        let template =
            Template::try_from(template).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let special_tokens = self
            .tokenizer
            .get_added_tokens_decoder()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.content.to_owned()))
            .collect::<Vec<(u32, String)>>()
            .to_vec();

        let tp = TemplateProcessing::builder()
            .single(template)
            .special_tokens(special_tokens)
            .build()
            .map_err(|e| runtime_err("set_post_processor", e))?;
        self.tokenizer.with_post_processor(Some(tp));
        Ok(())
    }

    #[pyo3(signature = (pretty = false))]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        self.tokenizer
            .to_string(pretty)
            .map_err(|e| runtime_err("to_str", e))
    }

    #[pyo3(signature = (path, pretty = true))]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        self.tokenizer.save(path, pretty).map_err(|e| {
            if let Some(io) = e.downcast_ref::<std::io::Error>() {
                io_err("save", std::io::Error::new(io.kind(), io.to_string()))
            } else {
                runtime_err("save", e)
            }
        })?;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| map_tok_err("from_file", e))?;
        Ok(Self::new(tokenizer))
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(with_added_tokens)
    }

    fn get_added_tokens_decoder(&self) -> HashMap<u32, String> {
        let mut added: HashMap<u32, String> = HashMap::new();
        for (id, token) in self.tokenizer.get_added_tokens_decoder().iter() {
            added.insert(*id, token.content.to_owned());
        }
        added
    }

    #[pyo3(signature = (input, with_added_tokens=true))]
    fn tokenize(&self, input: String, with_added_tokens: bool) -> PyResult<Vec<String>> {
        Ok(
            self.tokenizer
                .encode(input, with_added_tokens)
                .map_err(|e| map_tok_err("tokenize", e))?
                .get_tokens()
                .to_vec(),
        )
    }

    fn id_to_token(&self, index: u32) -> Option<String> {
        self.tokenizer.id_to_token(index)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn no_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }

    #[pyo3(signature = (**kwargs))]
    fn with_padding(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut params = PaddingParams::default();
        let mut unknown = Vec::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key: &str = key.extract()?;
                match key {
                    "direction" => {
                        let value: &str = value.extract()?;
                        params.direction = match value {
                            "left" => Ok(PaddingDirection::Left),
                            "right" => Ok(PaddingDirection::Right),
                            other => Err(value_err(
                                "with_padding",
                                format!("Unknown direction {other}"),
                            )),
                        }?
                    }
                    "pad_to_multiple_of" => params.pad_to_multiple_of = value.extract()?,
                    "pad_id" => params.pad_id = value.extract()?,
                    "pad_type_id" => params.pad_type_id = value.extract()?,
                    "pad_token" => params.pad_token = value.extract()?,
                    "length" => {
                        params.strategy = match value.extract()? {
                            Some(l) => PaddingStrategy::Fixed(l),
                            _ => PaddingStrategy::BatchLongest,
                        }
                    }
                    _ => unknown.push(key.to_string()),
                }
            }
        }
        if !unknown.is_empty() {
            return Err(type_err(
                "with_padding",
                format!(
                    "Unexpected keyword arguments: {} (allowed: direction, pad_to_multiple_of, pad_id, pad_type_id, pad_token, length)",
                    unknown.join(", ")
                ),
            ));
        }
        self.tokenizer.with_padding(Some(params));
        Ok(())
    }

    #[pyo3(signature = (**kwargs))]
    fn with_truncation(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut params = TruncationParams::default();
        let mut unknown = Vec::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "strategy" => {
                        let value: &str = value.extract()?;
                        params.strategy = match value {
                            "only_first" => Ok(TruncationStrategy::OnlyFirst),
                            "only_second" => Ok(TruncationStrategy::OnlySecond),
                            "longest_first" => Ok(TruncationStrategy::LongestFirst),
                            other => Err(value_err(
                                "with_truncation",
                                format!("Unknown truncation strategy {other}"),
                            )),
                        }?
                    }
                    "max_length" => params.max_length = value.extract()?,
                    "stride" => params.stride = value.extract()?,
                    "direction" => {
                        let value: &str = value.extract()?;
                        params.direction = match value {
                            "left" => Ok(TruncationDirection::Left),
                            "right" => Ok(TruncationDirection::Right),
                            other => Err(value_err(
                                "with_truncation",
                                format!("Unknown truncation direction {other}"),
                            )),
                        }?
                    }
                    _ => unknown.push(key.to_string()),
                }
            }
        }
        if !unknown.is_empty() {
            return Err(type_err(
                "with_truncation",
                format!(
                    "Unexpected keyword arguments: {} (allowed: strategy, max_length, stride, direction)",
                    unknown.join(", ")
                ),
            ));
        }
        let _ = self.tokenizer.with_truncation(Some(params));
        Ok(())
    }

    fn no_truncation(&mut self) -> PyResult<()> {
        let _ = self.tokenizer.with_truncation(None);
        Ok(())
    }

    fn add_tokens(&mut self, tokens: Vec<Bound<'_, PyAny>>) -> PyResult<usize> {
        let tokens: Vec<AddedToken> = tokens
            .into_iter()
            .map(|kwargs| {
                Ok(AddedToken {
                    content: kwargs.getattr("content")?.extract::<String>()?,
                    single_word: kwargs.getattr("single_word")?.extract::<bool>()?,
                    lstrip: kwargs.getattr("lstrip")?.extract::<bool>()?,
                    rstrip: kwargs.getattr("rstrip")?.extract::<bool>()?,
                    normalized: kwargs.getattr("normalized")?.extract::<bool>()?,
                    special: kwargs.getattr("special")?.extract::<bool>()?,
                })
            })
            .collect::<Result<Vec<AddedToken>, PyErr>>()?;
        Ok(self.tokenizer.add_tokens(&tokens))
    }

    #[pyo3(signature = (files, **kwargs))]
    fn train(
        &self,
        py: Python,
        files: Vec<String>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        // Construct Trainable Tokenizer
        let model: ModelWrapper = match self.tokenizer.get_model() {
            ModelWrapper::ModelWrapper(mw) => match mw {
                tokenizers::ModelWrapper::WordLevel(wl) => Ok(GPE::from(wl.to_owned())),
                _ => Err(runtime_err("train", "Unsupported model for training")),
            },
            ModelWrapper::GPE(gpe) => Ok(gpe.to_owned()),
        }?
        .into();

        // Remove any special tokens (i.e. [PAD]) from the initial vocab
        let is_special = Regex::new(r"\[[A-Z]+?\]").map_err(|e| runtime_err("train", e))?;
        let alphabet: HashSet<String> = self
            .tokenizer
            .get_vocab(false)
            .keys()
            .filter_map(|g| {
                if is_special.is_match(g) {
                    None
                } else {
                    Some(g.into())
                }
            })
            .collect();

        // Configure the trainer
        let mut builder = GpeTrainer::builder();
        let mut opt_split_structure = false;
        builder.alphabet(alphabet);
        let mut unknown = Vec::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key: &str = key.extract()?;
                match key {
                    "min_frequency" => {
                        builder.min_frequency(value.extract()?);
                    }
                    "vocab_size" => {
                        builder.vocab_size(value.extract()?);
                    }
                    "limit_alphabet" => {
                        builder.limit_alphabet(value.extract()?);
                    }
                    "merge_brackets" => {
                        builder.merge_brackets(value.extract()?);
                    }
                    "split_structure" => {
                        opt_split_structure = value.extract()?;
                    }
                    _ => unknown.push(key.to_string()),
                }
            }
        }
        if !unknown.is_empty() {
            return Err(type_err(
                "train",
                format!(
                    "Unexpected keyword arguments: {} (allowed: min_frequency, vocab_size, limit_alphabet, merge_brackets, split_structure)",
                    unknown.join(", ")
                ),
            ));
        }

        // Build the Smirk-GPE tokenizer
        let mut tok_builder = TokenizerBuilder::default()
            .with_normalizer(self.tokenizer.get_normalizer().cloned())
            .with_model(model)
            .with_decoder(self.tokenizer.get_decoder().cloned());

        if opt_split_structure {
            tok_builder = tok_builder.with_pre_tokenizer(Some(split_structure().into()));
        } else {
            tok_builder = tok_builder.with_pre_tokenizer(None);
        }
        let mut tokenizer: TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = tok_builder
            .build()
            .map_err(|e| runtime_err("train", e))?;

        // Train tokenizer
        let mut trainer: TrainerWrapper = builder.build().map_err(|e| runtime_err("train", e))?.into();
        let train_result = py.allow_threads(|| tokenizer.train_from_files(&mut trainer, files));
        train_result.map_err(|e| {
            if let Some(io) = e.downcast_ref::<std::io::Error>() {
                io_err("train", std::io::Error::new(io.kind(), io.to_string()))
            } else {
                runtime_err("train", e)
            }
        })?;
        Ok(SmirkTokenizer::new(tokenizer))
    }
}

#[derive(IntoPyObject, IntoPyObjectRef)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    pub offsets: Vec<(u64, u64)>,
}

impl From<tokenizers::Encoding> for Encoding {
    fn from(encoding: tokenizers::Encoding) -> Self {
        Self {
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
            offsets: encoding
                .get_offsets()
                .into_iter()
                .map(|&(start, end)| (start as u64, end as u64))
                .collect(),
        }
    }
}
