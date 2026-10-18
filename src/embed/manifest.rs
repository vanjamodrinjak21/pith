//! Compiled-in model manifest, pinned to an immutable Hugging Face commit.

/// One file to download: `remote` is relative to the manifest base URL, `local` is the file name
/// inside the model directory.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile<'a> {
    pub remote: &'a str,
    pub local: &'a str,
    pub sha256: &'a str,
    pub size: u64,
}

#[derive(Debug)]
pub struct Manifest {
    /// Model id recorded in databases; also the directory name under `models_dir()`.
    pub id: &'static str,
    pub dim: usize,
    /// `https://huggingface.co/<repo>/resolve/<commit sha>`.
    pub base_url: &'static str,
    pub license: &'static str,
    /// `[model, tokenizer]`.
    pub files: [ModelFile<'static>; 2],
}

pub const DEFAULT_MODEL: Manifest = Manifest {
    id: "bge-small-en-v1.5-int8",
    dim: 384,
    base_url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3",
    license: "MIT (derived from BAAI/bge-small-en-v1.5)",
    files: [
        ModelFile {
            remote: "onnx/model_quantized.onnx",
            local: "model_quantized.onnx",
            sha256: "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
            size: 34_014_426,
        },
        ModelFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            size: 711_396,
        },
    ],
};
