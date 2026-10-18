// Needs `test-hooks` (enabled for tests by the self dev-dependency in Cargo.toml).
#![cfg(all(feature = "mcp", feature = "test-hooks", debug_assertions))]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Session {
    fn start(project: &Path, home: &Path, fake: bool) -> Session {
        let mut c = Command::new(env!("CARGO_BIN_EXE_pith"));
        c.args(["mcp", "--project"])
            .arg(project)
            .env("PITH_HOME", home)
            .env_remove("PITH_TEST_FAKE_EMBEDDER")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if fake {
            c.env("PITH_TEST_FAKE_EMBEDDER", "1");
        }
        let mut child = c.spawn().expect("spawn pith mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut s = Session {
            child,
            stdin: Some(stdin),
            stdout,
            next_id: 1,
        };
        let r = s.request(
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}}),
        );
        assert_eq!(r["result"]["serverInfo"]["name"], "pith", "{r}");
        s.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        s
    }

    fn send(&mut self, v: Value) {
        let w = self.stdin.as_mut().unwrap();
        writeln!(w, "{v}").unwrap();
        w.flush().unwrap();
    }

    /// Every stdout line must be a JSON-RPC message: nothing else may print there.
    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "server closed"
            );
            let v: Value = serde_json::from_str(&line).expect("stdout line is JSON");
            if v["id"] == id {
                return v;
            }
        }
    }

    /// (is_error, text) of a tools/call.
    fn call(&mut self, name: &str, args: Value) -> (bool, String) {
        let r = self.request("tools/call", json!({"name": name, "arguments": args}));
        let res = &r["result"];
        assert!(res.is_object(), "{r}");
        let text = res["content"][0]["text"].as_str().unwrap_or_default();
        (res["isError"] == true, text.to_string())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn fixture() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("lib.rs"), "fn alpha() {\n    1\n}\n").unwrap();
    pithdb::index_project(d.path(), &pithdb::embed::FakeEmbedder, false).unwrap();
    d
}

#[test]
fn stdio_session_lists_three_tools_and_searches() {
    let d = fixture();
    let mut s = Session::start(d.path(), &d.path().join("home"), true);
    let r = s.request("tools/list", json!({}));
    let mut names: Vec<&str> = r["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["get_chunk", "search_code", "stats"]);

    let q = "fn alpha() {\n    1\n}";
    let (err, text) = s.call("search_code", json!({"query": q, "k": 3}));
    assert!(!err, "{text}");
    assert!(text.contains("lib.rs:1-3"), "{text}");
    assert!(text.contains("indexed "), "staleness missing: {text}");
    assert!(text.contains("fn alpha()"), "{text}");

    let (err, text) = s.call("get_chunk", json!({"chunk_id": 0}));
    assert!(!err && text.contains("fn alpha()"), "{text}");
    let (err, text) = s.call("stats", json!({}));
    assert!(!err && text.contains("fake-384"), "{text}");
    // Boundary validation.
    let (err, text) = s.call("search_code", json!({"query": "  "}));
    assert!(err, "{text}");
    let (err, text) = s.call("search_code", json!({"query": "x".repeat(5000)}));
    assert!(err, "{text}");
}

#[test]
fn closing_stdin_ends_the_session_cleanly() {
    let d = fixture();
    let mut s = Session::start(d.path(), &d.path().join("home"), true);
    drop(s.stdin.take()); // client hangs up
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let status = loop {
        if let Some(st) = s.child.try_wait().unwrap() {
            break st;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "server did not exit on EOF"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    assert!(status.success(), "{status}");
}

#[test]
fn get_chunk_unknown_id_is_a_tool_error_not_a_crash() {
    let d = fixture();
    let mut s = Session::start(d.path(), &d.path().join("home"), true);
    let (err, text) = s.call("get_chunk", json!({"chunk_id": 999}));
    assert!(err && text.contains("999"), "{text}");
    let (err, _) = s.call("stats", json!({}));
    assert!(!err, "server still serves after the error");
}

#[test]
fn no_project_found_reported_not_guessed() {
    let d = tempfile::tempdir().unwrap();
    let empty = d.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    let mut s = Session::start(&empty, &d.path().join("home"), true);
    for (tool, args) in [
        ("stats", json!({})),
        ("search_code", json!({"query": "x"})),
        ("get_chunk", json!({"chunk_id": 0})),
    ] {
        let (err, text) = s.call(tool, args);
        assert!(err && text.contains("no Pith index"), "{tool}: {text}");
    }
}

#[cfg(feature = "onnx")]
#[test]
fn search_code_with_no_model_says_run_pith_init() {
    let d = fixture();
    let mut s = Session::start(d.path(), &d.path().join("home"), false);
    let (err, text) = s.call("search_code", json!({"query": "alpha"}));
    assert!(err && text.contains("run `pith init`"), "{text}");
}

#[test]
fn overlong_stdin_line_ends_the_session() {
    let d = fixture();
    let mut s = Session::start(d.path(), &d.path().join("home"), true);
    // Write from a thread: a regressed cap would block the pipe instead of failing the test.
    let mut stdin = s.stdin.take().unwrap();
    std::thread::spawn(move || {
        let _ = stdin.write_all(&vec![b'x'; 2 << 20]);
        let _ = stdin.flush();
        std::thread::sleep(std::time::Duration::from_secs(60)); // keep stdin open
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let status = loop {
        if let Some(st) = s.child.try_wait().unwrap() {
            break st;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "server did not end the session on an overlong line"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    // The read error ends the session and `pith mcp` exits with an error status.
    assert!(!status.success(), "{status}");
}
