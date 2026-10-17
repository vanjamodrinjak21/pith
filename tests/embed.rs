use pithdb::embed::{
    cls_pool_normalize, pull_files, Embedder, FakeEmbedder, ModelFile, DEFAULT_MODEL,
};
use pithdb::Error;
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// What the test server sends for one request.
#[derive(Clone)]
enum Reply {
    /// 200 with this body and a Content-Length header (overridable).
    Body(Vec<u8>, Option<usize>),
    /// 200 without Content-Length, closing after the body.
    Unsized(Vec<u8>),
    /// 302 to this location.
    Redirect(String),
}

/// Minimal HTTP/1.1 server: maps request path to a reply; 404 otherwise. Serves until dropped.
fn serve(routes: Vec<(&'static str, Reply)>) -> String {
    serve_logged(routes).0
}

/// Like [`serve`], also returning the log of requested paths.
fn serve_logged(routes: Vec<(&'static str, Reply)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log2 = log.clone();
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            if r.read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            log2.lock().unwrap().push(path.clone());
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                    break;
                }
            }
            let reply = routes
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, r)| r.clone());
            let _ = match reply {
                Some(Reply::Body(b, len)) => {
                    let len = len.unwrap_or(b.len());
                    let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n");
                    s.write_all(&b)
                }
                Some(Reply::Unsized(b)) => {
                    let _ = write!(s, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n");
                    s.write_all(&b)
                }
                Some(Reply::Redirect(loc)) => write!(
                    s,
                    "HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                ),
                None => write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
            };
            let _ = s.flush();
            let _ = s.shutdown(std::net::Shutdown::Both);
            let mut sink = Vec::new();
            let _ = r.read_to_end(&mut sink);
        }
    });
    (format!("http://{addr}"), log)
}

struct Fixture {
    a: Vec<u8>,
    b: Vec<u8>,
    sha_a: String,
    sha_b: String,
}

impl Fixture {
    fn new() -> Self {
        let a = b"model bytes ".repeat(1000);
        let b = b"{\"tokenizer\": true}".to_vec();
        Fixture {
            sha_a: hex(&a),
            sha_b: hex(&b),
            a,
            b,
        }
    }
    fn files(&self) -> [ModelFile<'_>; 2] {
        [
            ModelFile {
                remote: "onnx/m.onnx",
                local: "m.onnx",
                sha256: &self.sha_a,
                size: self.a.len() as u64,
            },
            ModelFile {
                remote: "tok.json",
                local: "tok.json",
                sha256: &self.sha_b,
                size: self.b.len() as u64,
            },
        ]
    }
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|r| {
            r.map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect()
        })
        .unwrap_or_default();
    v.retain(|n| n != ".pull.lock"); // the (empty) lock file is expected to stay
    v.sort();
    v
}

#[test]
fn fake_embedder_is_normalized_and_deterministic() {
    let e = FakeEmbedder;
    assert_eq!(e.model_id(), "fake-384");
    assert_eq!(e.dim(), 384);
    let texts = vec!["fn main() {}".to_string(), String::new(), "x".repeat(5000)];
    let a = e.embed_passages(&texts).unwrap();
    let b = e.embed_passages(&texts).unwrap();
    assert_eq!(a, b);
    for v in &a {
        assert_eq!(v.len(), 384);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!((norm(v) - 1.0).abs() < 1e-5);
    }
    assert_ne!(a[0], a[1]);
    assert_eq!(e.embed_query("fn main() {}").unwrap(), a[0]);
}

#[test]
fn cls_pooling_then_normalize_matches_reference_vector() {
    // batch 2, seq 2, dim 2: CLS rows are [3,4] and [0,-2]; the second token is ignored.
    let hidden = [3.0, 4.0, 9.0, 9.0, 0.0, -2.0, 7.0, 7.0];
    let out = cls_pool_normalize(&hidden, 2, 2, 2).unwrap();
    assert_eq!(out, vec![vec![0.6, 0.8], vec![0.0, -1.0]]);
    assert!(matches!(
        cls_pool_normalize(&[f32::NAN, 1.0], 1, 1, 2),
        Err(Error::NonFinite)
    ));
    assert!(matches!(
        cls_pool_normalize(&[f32::INFINITY, 1.0], 1, 1, 2),
        Err(Error::NonFinite)
    ));
    assert!(matches!(
        cls_pool_normalize(&[0.0, 0.0], 1, 1, 2),
        Err(Error::EmptyVector)
    ));
    assert!(cls_pool_normalize(&[1.0; 3], 1, 1, 2).is_err());
}

#[test]
fn pull_succeeds_with_matching_hash() {
    let f = Fixture::new();
    let base = serve(vec![
        ("/onnx/m.onnx", Reply::Body(f.a.clone(), None)),
        // exercise a relative redirect, like the Hugging Face resolve endpoint
        (
            "/tok.json",
            Reply::Redirect("/cdn/tok.json?sig=secret".into()),
        ),
        ("/cdn/tok.json?sig=secret", Reply::Body(f.b.clone(), None)),
    ]);
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("models/x");
    pull_files(&dir, &base, &f.files()).unwrap();
    assert_eq!(std::fs::read(dir.join("m.onnx")).unwrap(), f.a);
    assert_eq!(std::fs::read(dir.join("tok.json")).unwrap(), f.b);
    assert_eq!(names(&dir), ["m.onnx", "tok.json"]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }
    // second pull with files already in place needs no network at all
    pull_files(&dir, "http://127.0.0.1:1", &f.files()).unwrap();
}

#[test]
fn pull_rejects_hash_mismatch_and_writes_nothing() {
    let f = Fixture::new();
    let mut wrong = f.a.clone();
    wrong[0] ^= 1;
    let base = serve(vec![
        ("/onnx/m.onnx", Reply::Body(wrong, None)),
        ("/tok.json", Reply::Body(f.b.clone(), None)),
    ]);
    let d = tempfile::tempdir().unwrap();
    let err = pull_files(d.path(), &base, &f.files()).unwrap_err();
    assert!(matches!(err, Error::Integrity(_)), "{err}");
    assert!(names(d.path()).is_empty(), "{:?}", names(d.path()));
}

#[test]
fn pull_rejects_size_mismatch_and_oversize_stream() {
    let f = Fixture::new();
    let short = f.a[..f.a.len() - 1].to_vec();
    let mut long = f.a.clone();
    long.extend_from_slice(b"extra");
    let base = serve(vec![
        ("/short/onnx/m.onnx", Reply::Body(short, None)),
        (
            "/lie/onnx/m.onnx",
            Reply::Body(long.clone(), Some(f.a.len())),
        ),
        ("/unsized/onnx/m.onnx", Reply::Unsized(long.clone())),
        ("/big/onnx/m.onnx", Reply::Body(long, None)),
        (
            "/shortunsized/onnx/m.onnx",
            Reply::Unsized(f.a[..100].to_vec()),
        ),
        // honest-looking Content-Length, then the connection drops mid-body
        (
            "/midstream/onnx/m.onnx",
            Reply::Body(f.a[..f.a.len() / 2].to_vec(), Some(f.a.len())),
        ),
    ]);
    for p in ["short", "unsized", "big", "shortunsized", "midstream"] {
        let d = tempfile::tempdir().unwrap();
        let err = pull_files(d.path(), &format!("{base}/{p}"), &f.files()).unwrap_err();
        assert!(matches!(err, Error::Download(_)), "{p}: {err}");
        assert!(names(d.path()).is_empty(), "{p}");
    }
    // Content-Length matches the manifest but trailing bytes follow: only the declared bytes are read.
    let d = tempfile::tempdir().unwrap();
    pull_files(d.path(), &format!("{base}/lie"), &f.files()[..1]).unwrap();
    assert_eq!(std::fs::read(d.path().join("m.onnx")).unwrap(), f.a);
}

#[test]
fn pull_rejects_plain_http_to_non_loopback_and_https_downgrade() {
    let f = Fixture::new();
    let d = tempfile::tempdir().unwrap();
    for base in [
        "http://example.com/x",
        "ftp://127.0.0.1/x",
        "http://127.0.0.1.evil.com/x",
        "http://user@127.0.0.1/x",
    ] {
        let err = pull_files(d.path(), base, &f.files()).unwrap_err();
        assert!(matches!(err, Error::Download(_)), "{base}: {err}");
    }
    // End to end through download(): redirects off loopback over plain http are refused before
    // any connection is made to the target.
    // A refused redirect makes no request to its target: point it at a second logged server.
    let (target, target_log) = serve_logged(vec![]);
    let target_port = target.rsplit(':').next().unwrap();
    for to in [
        "http://example.com/m.onnx".to_string(),
        "http://192.168.1.1/m.onnx".to_string(),
        "https://example.com/m.onnx".to_string(),
        format!("ftp://127.0.0.1:{target_port}/m.onnx"),
    ] {
        let base = serve(vec![("/onnx/m.onnx", Reply::Redirect(to.clone()))]);
        let err = pull_files(d.path(), &base, &f.files()).unwrap_err();
        assert!(err.to_string().contains("refusing redirect"), "{to}: {err}");
        assert!(names(d.path()).is_empty());
    }
    assert!(target_log.lock().unwrap().is_empty());
}

#[test]
fn failed_pull_leaves_existing_model_intact() {
    let f = Fixture::new();
    let d = tempfile::tempdir().unwrap();
    // existing good model on disk, but the manifest now wants different bytes for file b
    std::fs::write(d.path().join("m.onnx"), b"old good model").unwrap();
    std::fs::write(d.path().join("tok.json"), b"old tokenizer").unwrap();
    let base = serve(vec![
        ("/onnx/m.onnx", Reply::Body(f.a.clone(), None)),
        (
            "/tok.json",
            Reply::Body(b"{\"tokenizer\": tru!}".to_vec(), None),
        ),
    ]);
    pull_files(d.path(), &base, &f.files()).unwrap_err();
    assert_eq!(
        std::fs::read(d.path().join("m.onnx")).unwrap(),
        b"old good model"
    );
    assert_eq!(
        std::fs::read(d.path().join("tok.json")).unwrap(),
        b"old tokenizer"
    );
    assert_eq!(names(d.path()), ["m.onnx", "tok.json"]);
}

#[test]
fn pull_removes_stale_temp_files() {
    let f = Fixture::new();
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join(".pull-crashed"), b"partial").unwrap();
    std::fs::write(d.path().join("keep.txt"), b"unrelated").unwrap();
    let base = serve(vec![
        ("/onnx/m.onnx", Reply::Body(f.a.clone(), None)),
        ("/tok.json", Reply::Body(f.b.clone(), None)),
    ]);
    pull_files(d.path(), &base, &f.files()).unwrap();
    assert_eq!(names(d.path()), ["keep.txt", "m.onnx", "tok.json"]);
}

#[test]
fn default_manifest_is_pinned() {
    let m = &DEFAULT_MODEL;
    assert_eq!(m.dim, 384);
    assert!(m
        .base_url
        .starts_with("https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/"));
    assert_eq!(
        m.base_url.rsplit('/').next().unwrap().len(),
        40,
        "revision must be a commit sha"
    );
    assert!(m.license.starts_with("MIT"));
    for f in &m.files {
        assert_eq!(f.sha256.len(), 64);
        assert!(f.size > 0);
    }
}

#[test]
fn models_dir_honours_pith_home() {
    // Only this test touches PITH_HOME in this binary.
    std::env::set_var("PITH_HOME", "/tmp/pith-home-test");
    assert_eq!(
        pithdb::embed::models_dir().unwrap(),
        Path::new("/tmp/pith-home-test/models")
    );
    std::env::remove_var("PITH_HOME");
    let home = std::env::var_os("HOME").unwrap();
    assert_eq!(
        pithdb::embed::models_dir().unwrap(),
        Path::new(&home).join(".pith/models")
    );
    // never a relative path
    std::env::set_var("HOME", "relative/home");
    assert!(pithdb::embed::models_dir().is_err());
    std::env::remove_var("HOME");
    assert!(pithdb::embed::models_dir().is_err());
    std::env::set_var("HOME", &home);
}

#[test]
fn normalize_survives_huge_values() {
    let out = cls_pool_normalize(&[3e30, 4e30], 1, 1, 2).unwrap();
    assert!(
        (out[0][0] - 0.6).abs() < 1e-6 && (out[0][1] - 0.8).abs() < 1e-6,
        "{out:?}"
    );
    assert!(cls_pool_normalize(&[1.0; 4], usize::MAX, 2, 2).is_err());
}

#[test]
fn pull_rejects_unsafe_local_names() {
    let f = Fixture::new();
    let d = tempfile::tempdir().unwrap();
    for local in ["../escape", "a/b", "/abs", ".."] {
        let mut files = f.files();
        files[0].local = local;
        let err = pull_files(&d.path().join("m"), "http://127.0.0.1:1", &files).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{local}: {err}");
    }
    assert!(names(d.path()).is_empty());
}

#[test]
fn concurrent_pulls_serialize_and_both_succeed() {
    let f = Fixture::new();
    let (base, log) = serve_logged(vec![
        ("/onnx/m.onnx", Reply::Body(f.a.clone(), None)),
        ("/tok.json", Reply::Body(f.b.clone(), None)),
    ]);
    let d = tempfile::tempdir().unwrap();
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..4)
            .map(|_| s.spawn(|| pull_files(d.path(), &base, &f.files())))
            .collect();
        for h in hs {
            h.join().unwrap().unwrap();
        }
    });
    assert_eq!(names(d.path()), ["m.onnx", "tok.json"]);
    assert!(d.path().join(".pull.lock").exists());
    assert_eq!(std::fs::read(d.path().join("m.onnx")).unwrap(), f.a);
    // With the lock, later pulls find verified files: each file is fetched exactly once.
    let mut got = log.lock().unwrap().clone();
    got.sort();
    assert_eq!(got, ["/onnx/m.onnx", "/tok.json"]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(d.path().join(".pull.lock"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[cfg(unix)]
#[test]
fn pull_keeps_existing_dir_mode_and_refuses_symlinked_dir() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let base = serve(vec![
        ("/onnx/m.onnx", Reply::Body(f.a.clone(), None)),
        ("/tok.json", Reply::Body(f.b.clone(), None)),
    ]);
    let d = tempfile::tempdir().unwrap();
    let existing = d.path().join("existing");
    std::fs::create_dir(&existing).unwrap();
    std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o750)).unwrap();
    pull_files(&existing, &base, &f.files()).unwrap();
    let mode = std::fs::metadata(&existing).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o750);
    let link = d.path().join("link");
    std::os::unix::fs::symlink(&existing, &link).unwrap();
    let err = pull_files(&link, &base, &f.files()).unwrap_err();
    assert!(matches!(err, Error::Invalid(_)), "{err}");
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use pithdb::embed::{OnnxConfig, OnnxEmbedder};

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn load_without_model_returns_model_missing() {
        let d = tempfile::tempdir().unwrap();
        let err = OnnxEmbedder::load(d.path()).err().expect("must fail");
        assert!(matches!(err, Error::ModelMissing(_)), "{err}");
        assert!(err.to_string().contains("pith model pull"));
    }

    #[test]
    fn load_rejects_truncated_model() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join(DEFAULT_MODEL.id);
        std::fs::create_dir_all(&dir).unwrap();
        for f in &DEFAULT_MODEL.files {
            std::fs::write(dir.join(f.local), b"short").unwrap();
        }
        let err = OnnxEmbedder::load(d.path()).err().expect("must fail");
        assert!(matches!(err, Error::Integrity(_)), "{err}");
    }

    #[test]
    fn load_rejects_right_size_wrong_hash() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join(DEFAULT_MODEL.id);
        std::fs::create_dir_all(&dir).unwrap();
        for f in &DEFAULT_MODEL.files {
            std::fs::File::create(dir.join(f.local))
                .unwrap()
                .set_len(f.size)
                .unwrap();
        }
        let err = OnnxEmbedder::load(d.path()).err().expect("must fail");
        assert!(matches!(err, Error::Integrity(_)), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn load_refuses_symlinked_model_files() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join(DEFAULT_MODEL.id);
        std::fs::create_dir_all(&dir).unwrap();
        for f in &DEFAULT_MODEL.files {
            let target = d.path().join(format!("real-{}", f.local));
            std::fs::File::create(&target)
                .unwrap()
                .set_len(f.size)
                .unwrap();
            std::os::unix::fs::symlink(&target, dir.join(f.local)).unwrap();
        }
        let err = OnnxEmbedder::load(d.path()).err().expect("must fail");
        assert!(matches!(err, Error::Integrity(_)), "{err}");
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn load_refuses_symlinked_model_dir() {
        let d = tempfile::tempdir().unwrap();
        let real_dir = d.path().join("elsewhere");
        std::fs::create_dir_all(&real_dir).unwrap();
        for f in &DEFAULT_MODEL.files {
            std::fs::File::create(real_dir.join(f.local))
                .unwrap()
                .set_len(f.size)
                .unwrap();
        }
        std::os::unix::fs::symlink(&real_dir, d.path().join(DEFAULT_MODEL.id)).unwrap();
        let err = OnnxEmbedder::load(d.path()).err().expect("must fail");
        assert!(matches!(err, Error::Integrity(_)), "{err}");
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    fn real() -> OnnxEmbedder {
        OnnxEmbedder::load(&pithdb::embed::models_dir().unwrap())
            .expect("run the pull first (see report)")
    }

    #[test]
    #[ignore = "needs the pulled model"]
    fn real_model_ranks_related_text_higher() {
        let e = real();
        let q = e
            .embed_query("how do I open a file and read it into a string")
            .unwrap();
        let p = e
            .embed_passages(&[
                "fn read_file(path: &Path) -> io::Result<String> { std::fs::read_to_string(path) }"
                    .into(),
                "The cat sat on the warm windowsill watching birds.".into(),
            ])
            .unwrap();
        let (rel, unrel) = (dot(&q, &p[0]), dot(&q, &p[1]));
        println!("related {rel:.4} unrelated {unrel:.4}");
        assert!(rel > unrel);
    }

    #[test]
    #[ignore = "needs the pulled model"]
    fn real_model_handles_edge_inputs_and_batch_order() {
        let e = real();
        let texts: Vec<String> = vec![
            "short".into(),
            String::new(),
            "word ".repeat(3000), // far beyond 512 tokens: truncated, not an error
            "fn add(a: i32, b: i32) -> i32 { a + b }".into(),
            "a medium length sentence about vector databases and embeddings".into(),
        ];
        let batched = e.embed_passages(&texts).unwrap();
        for (t, v) in texts.iter().zip(&batched) {
            assert_eq!(v.len(), 384);
            assert!(v.iter().all(|x| x.is_finite()));
            assert!((norm(v) - 1.0).abs() < 1e-5);
            let alone = e.embed_passages(std::slice::from_ref(t)).unwrap();
            let diff = alone[0]
                .iter()
                .zip(v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(diff < 1e-4, "default config: list vs alone diff {diff}");
        }
        for bs in [1, 2, 32] {
            let e2 = OnnxEmbedder::load_with(
                &pithdb::embed::models_dir().unwrap(),
                OnnxConfig {
                    batch_size: bs,
                    ..OnnxConfig::default()
                },
            )
            .unwrap();
            let v = e2.embed_passages(&texts).unwrap();
            // Dynamic int8 quantization shares activation scales across a batch, so batch > 1 only
            // approximates batch 1; it must stay close, never reorder or mix up rows.
            for (a, b) in v.iter().zip(&batched) {
                let cos = dot(a, b);
                println!("batch size {bs}: cosine to batch-1 result {cos:.6}");
                assert!(cos > 0.99, "batch size {bs}: cosine {cos}");
            }
        }
    }

    #[test]
    #[ignore = "needs the pulled model"]
    fn real_model_throughput() {
        for (bs, th) in [(1usize, 1usize), (1, 4), (1, 8), (32, 8)] {
            let e = OnnxEmbedder::load_with(
                &pithdb::embed::models_dir().unwrap(),
                OnnxConfig {
                    batch_size: bs,
                    threads: th,
                },
            )
            .unwrap();
            let chunk =
                "pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> { /* scan */ }\n".repeat(12);
            let texts: Vec<String> = (0..256).map(|i| format!("{i} {chunk}")).collect();
            let t = std::time::Instant::now();
            let v = e.embed_passages(&texts).unwrap();
            let s = t.elapsed().as_secs_f64();
            assert_eq!(v.len(), 256);
            assert!(v
                .iter()
                .all(|x| x.iter().all(|f| f.is_finite()) && (norm(x) - 1.0).abs() < 1e-5));
            assert!(
                256.0 / s >= 10.0, // floor far below the measured ~45/s; run with --release
                "throughput collapsed: {:.2} chunks/s",
                256.0 / s
            );
            println!(
                "batch {bs} threads {th}: {:.1} chunks/s ({} chunks of {} chars)",
                256.0 / s,
                texts.len(),
                texts[0].len()
            );
        }
    }
}

#[test]
#[ignore = "downloads ~35 MB from Hugging Face into models_dir()"]
fn pull_default_model_from_hugging_face() {
    let dir = pithdb::embed::models_dir().unwrap();
    pithdb::embed::pull(&dir, DEFAULT_MODEL.base_url).unwrap();
    pithdb::embed::pull(&dir, DEFAULT_MODEL.base_url).unwrap(); // idempotent, no re-download
}
