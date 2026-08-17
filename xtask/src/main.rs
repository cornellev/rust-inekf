use anyhow::{bail, Context, Result};
use std::{env, fs, path::{Path, PathBuf}, process::Command};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn run(cmd: &mut Command) -> Result<()> {
    let status = cmd.status().with_context(|| format!("could not spawn {cmd:?}"))?;
    if !status.success() {
        bail!("{cmd:?} failed with {status}");
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &dest)?;
        } else {
            fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

fn docs(open: bool) -> Result<()> {
    let root = root();
    let header = fs::canonicalize(root.join("docs/katex-header.html"))
        .context("docs/katex-header.html not found")?;
    // rustdoc resolves --html-in-header against the CWD, not the manifest.
    // so this path must be absolute.
    run(Command::new("cargo")
        .current_dir(&root)
        .args(["doc", "--no-deps", "-p", "rust-inekf"])
        .env("RUSTDOCFLAGS", format!("--html-in-header {}", header.display())))?;

    run(Command::new("mdbook")
        .current_dir(&root)
        .args(["build", "docs"]))?;

    // Nest the API reference inside the book so one directory is deployable.
    copy_dir(&root.join("target/doc"), &root.join("docs/book/api"))?;

    if open {
        let index = root.join("docs/book/index.html");
        run(Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" })
            .arg(index))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("docs") => docs(args.iter().any(|a| a == "--open")),
        _ => {
            eprintln!("usage: cargo docs [--open]");
            Ok(())
        }
    }
}
