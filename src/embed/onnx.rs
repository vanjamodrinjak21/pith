//! bge-small-en-v1.5 (int8) via ONNX Runtime: CLS pooling, L2 normalization, 512-token truncation.

use super::manifest::DEFAULT_MODEL;
use super::pull::sha256_hex;
use super::{cls_pool_normalize, Embedder};
use crate::{Error, Result};
use ort::environment::Environment;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{Tokenizer, TruncationParams};

const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
const MAX_TOKENS: usize = 512;

/// `batch_size` defaults to 1: the int8 model quantizes activations dynamically with scales shared
/// across the whole batch, so batched results depend on batch neighbours (measured cosine to the
/// batch-1 vector ~0.996-0.9995), and on this CPU batching gave no throughput gain. The batch size
/// is not part of `model_id`, so a database must be indexed and queried with the same value.
#[derive(Debug, Clone, Copy)]
pub struct OnnxConfig {
    pub batch_size: usize,
    pub threads: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        OnnxConfig {
            batch_size: 1,
            threads: cores.min(8),
        }
    }
}

/// Loaded model. Unload by dropping it.
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    cfg: OnnxConfig,
    token_types: bool,
}

fn oe(e: impl std::fmt::Display) -> Error {
    Error::Embed(e.to_string())
}

/// ORT allows building one environment per process, ever (even after drop), so it is kept for the
/// process lifetime; it is small. Sessions (the big part) are freed when the embedder drops.
static ENV: Mutex<Option<Environment>> = Mutex::new(None);

fn environment() -> Result<Environment> {
    let mut g = ENV
        .lock()
        .map_err(|_| oe("ONNX environment lock poisoned"))?;
    if let Some(e) = &*g {
        return Ok(e.clone());
    }
    // Telemetry off: Pith never phones home (ORT only emits it on Windows, but be explicit).
    let e = ort::environment::init()
        .with_name("pith")
        .with_telemetry(false)
        .build()
        .map_err(oe)?;
    *g = Some(e.clone());
    Ok(e)
}

/// Reads a file without following a symlink at its final component.
fn read_no_follow(p: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        &mut o,
        rustix::fs::OFlags::NOFOLLOW.bits() as i32,
    );
    let mut data = Vec::new();
    o.open(p)?.read_to_end(&mut data)?;
    Ok(data)
}

impl OnnxEmbedder {
    pub fn load(models_dir: &Path) -> Result<Self> {
        Self::load_with(models_dir, OnnxConfig::default())
    }

    /// Vectors depend slightly on `cfg.batch_size` (see [`OnnxConfig`]): index and query with the
    /// same value. Verifies size, then sha256 of both files (~20 ms for the 34 MB model in a release build on
    /// an M4 Pro, measured), so a corrupted or swapped file is never loaded. Offline.
    pub fn load_with(models_dir: &Path, cfg: OnnxConfig) -> Result<Self> {
        let dir = models_dir.join(DEFAULT_MODEL.id);
        let mut bytes = Vec::new();
        for f in &DEFAULT_MODEL.files {
            let p = dir.join(f.local);
            match std::fs::symlink_metadata(&p) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(Error::ModelMissing(p.display().to_string()))
                }
                Err(e) => return Err(e.into()),
                Ok(m) if !m.file_type().is_file() => {
                    return Err(Error::Integrity(format!(
                        "{} is not a regular file (symlinks are refused); run `pith model pull`",
                        p.display()
                    )))
                }
                Ok(m) if m.len() != f.size => {
                    return Err(Error::Integrity(format!(
                        "{} is {} bytes, expected {}; run `pith model pull`",
                        p.display(),
                        m.len(),
                        f.size
                    )))
                }
                Ok(_) => {}
            }
            let data = read_no_follow(&p)?;
            if sha256_hex(&data) != f.sha256 {
                return Err(Error::Integrity(format!(
                    "{} has the wrong sha256; run `pith model pull`",
                    p.display()
                )));
            }
            bytes.push(data);
        }
        let mut tokenizer = Tokenizer::from_bytes(&bytes[1]).map_err(oe)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(oe)?;
        tokenizer.with_padding(None);
        let env = environment()?;
        let session = Session::builder(&env)
            .map_err(oe)?
            .with_intra_threads(cfg.threads.max(1))
            .map_err(oe)?
            .commit_from_memory(&bytes[0])
            .map_err(oe)?;
        let token_types = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        Ok(OnnxEmbedder {
            session: Mutex::new(session),
            tokenizer,
            cfg,
            token_types,
        })
    }

    fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let dim = DEFAULT_MODEL.dim;
        let enc = self.tokenizer.encode_batch(texts, true).map_err(oe)?;
        // Sort by length so each batch pads to similar lengths; results go back by index.
        let mut order: Vec<usize> = (0..enc.len()).collect();
        order.sort_by_key(|&i| enc[i].len());
        let mut out = vec![Vec::new(); enc.len()];
        let mut session = self
            .session
            .lock()
            .map_err(|_| oe("model session poisoned"))?;
        for idxs in order.chunks(self.cfg.batch_size.max(1)) {
            let (b, seq) = (
                idxs.len(),
                idxs.iter().map(|&i| enc[i].len()).max().unwrap_or(0),
            );
            let mut ids = vec![0i64; b * seq];
            let mut mask = vec![0i64; b * seq];
            let mut types = vec![0i64; b * seq];
            for (r, &i) in idxs.iter().enumerate() {
                let e = &enc[i];
                for (c, ((&id, &m), &t)) in e
                    .get_ids()
                    .iter()
                    .zip(e.get_attention_mask())
                    .zip(e.get_type_ids())
                    .enumerate()
                {
                    ids[r * seq + c] = id.into();
                    mask[r * seq + c] = m.into();
                    types[r * seq + c] = t.into();
                }
            }
            let shape = [b, seq];
            let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
                (
                    "input_ids".into(),
                    Tensor::from_array((shape, ids)).map_err(oe)?.into(),
                ),
                (
                    "attention_mask".into(),
                    Tensor::from_array((shape, mask)).map_err(oe)?.into(),
                ),
            ];
            if self.token_types {
                inputs.push((
                    "token_type_ids".into(),
                    Tensor::from_array((shape, types)).map_err(oe)?.into(),
                ));
            }
            let outs = session.run(inputs).map_err(oe)?;
            let hidden = outs
                .get("last_hidden_state")
                .ok_or_else(|| oe("model has no last_hidden_state output"))?;
            let (s, data) = hidden.try_extract_tensor::<f32>().map_err(oe)?;
            let dims: Vec<usize> = s.iter().filter_map(|&d| usize::try_from(d).ok()).collect();
            if dims.len() != 3 || s.len() != 3 || dims[0] != b || dims[2] != dim {
                return Err(oe(format!("unexpected output shape {s:?}")));
            }
            let vecs = cls_pool_normalize(data, b, dims[1], dim)?;
            for (&i, v) in idxs.iter().zip(vecs) {
                out[i] = v;
            }
        }
        Ok(out)
    }
}

impl Embedder for OnnxEmbedder {
    fn model_id(&self) -> &str {
        DEFAULT_MODEL.id
    }
    fn dim(&self) -> usize {
        DEFAULT_MODEL.dim
    }
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts.iter().map(String::as_str).collect())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let q = format!("{QUERY_PREFIX}{text}");
        Ok(self.embed(vec![q.as_str()])?.remove(0))
    }
}
