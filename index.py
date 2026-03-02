
# app/run.py
import os
import subprocess
from pathlib import Path
import yaml
import shutil
import requests, xml.etree.ElementTree as ET, html
import asyncio, inspect, threading, traceback

print("=== GraphRAG Index Job starting ===")

# -------------------------------
# 1) Paths (allow env override)
# -------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/lakehouse/default/Files/graphrag_project")).resolve()
INPUT_DIR   = PROJECT_ROOT / "input"
OUTPUT_DIR  = PROJECT_ROOT / "output"
for p in (PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

print(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[paths] INPUT_DIR={INPUT_DIR}")
print(f"[paths] OUTPUT_DIR={OUTPUT_DIR}")
env = os.environ.copy()

# -------------------------------
# 2) AOAI env
# -------------------------------
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT","https://kd-foundry-accelerators-prj.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHAT_DEPLOYMENT          = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT","gpt-4o")
EMB_DEPLOYMENT           = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT","text-embedding-3-large")

missing = [k for k,v in {
    "AZURE_OPENAI_API_KEY":AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT":AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_VERSION":AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_CHAT_DEPLOYMENT":CHAT_DEPLOYMENT,
    "AZURE_OPENAI_EMB_DEPLOYMENT":EMB_DEPLOYMENT,
}.items() if not v]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

# .env for GraphRAG
(PROJECT_ROOT / ".env").write_text(
    f"AZURE_OPENAI_API_KEY={AZURE_OPENAI_API_KEY}\n"
    f"AZURE_OPENAI_ENDPOINT={AZURE_OPENAI_ENDPOINT}\n"
    f"AZURE_OPENAI_API_VERSION={AZURE_OPENAI_API_VERSION}\n"
)
print(f"[env] .env written at {PROJECT_ROOT / '.env'}")

# -------------------------------
# 3) GraphRAG settings.yaml
#    If settings.yaml exists in PROJECT_ROOT, use it.
#    Otherwise generate one using env vars.
# -------------------------------
settings_path = PROJECT_ROOT / "settings.yaml"
if not settings_path.exists():
  settings = {
    "models": {
        "default_chat_model": {
            "type": "chat",
            "model_provider": "azure",
            "auth_type": "api_key",
            "api_key": "${AZURE_OPENAI_API_KEY}",         
	    "api_base": "${AZURE_OPENAI_ENDPOINT}",
	    "api_version": "${AZURE_OPENAI_API_VERSION}",
	    "deployment_name": CHAT_DEPLOYMENT, 
            "model": "gpt-4o",
            "model_supports_json": True,
            "concurrent_requests": 5,
            "async_mode": "threaded",
            "retry_strategy": "exponential_backoff",
            "max_retries": 10,
            "tokens_per_minute": 8000,
            "requests_per_minute": 60,
            "temperature": 0,
        },
        "default_embedding_model": {
            "type": "embedding",
            "model_provider": "azure",
            "auth_type": "api_key",
            "api_key": "${AZURE_OPENAI_API_KEY}",
            "api_base": "${AZURE_OPENAI_ENDPOINT}",
            "api_version": "${AZURE_OPENAI_API_VERSION}",
            "deployment_name": EMB_DEPLOYMENT,
            "model": "text-embedding-3-large",
            "concurrent_requests": 10,
            "async_mode": "threaded",
            "retry_strategy": "exponential_backoff",
            "max_retries": 10,
            "tokens_per_minute": 80000,
            "requests_per_minute": 120,
        },
    },

    "input": {
        "storage": {
            "type": "file",
            "base_dir": "input",
        },
        "file_type": "text",
        "file_pattern": r".*\.txt",
    },

    "chunks": {
        "size": 300,
        "overlap": 100,
        "group_by_columns": ["id"],
    },

    "output": {
        "type": "file",
        "base_dir": "output",
    },

    "cache": {
        "type": "file",
        "base_dir": "cache",
    },

    "reporting": {
        "type": "file",
        "base_dir": "logs",
    },

    "vector_store": {
        "default_vector_store": {
            "type": "lancedb",
            # pinned path like your snippet; adjust if you want to derive from PROJECT_ROOT
            "path": "/lakehouse/default/Files/graphrag_project/lancedb",
            "container_name": "default",
            "overwrite": True,
            "create_if_missing": True,
        }
    },

    "embed_text": {
        "model_id": "default_embedding_model",
        "vector_store_id": "default_vector_store",
    },

    "extract_graph": {
        "model_id": "default_chat_model",
        "prompt": "prompts/extract_graph.txt",
        "entity_types": ["organization", "person", "geo", "event"],
        "max_gleanings": 0,
    },

    "summarize_descriptions": {
        "model_id": "default_chat_model",
        "prompt": "prompts/summarize_descriptions.txt",
        "max_length": 500,
    },

    "extract_claims": {
        "enabled": False,
        "model_id": "default_chat_model",
        "prompt": "prompts/extract_claims.txt",
        "description": "Any claims or facts that could be relevant to information discovery.",
        "max_gleanings": 0,
    },

    "community_reports": {
        "model_id": "default_chat_model",
        "graph_prompt": "prompts/community_report_graph.txt",
        "text_prompt": "prompts/community_report_text.txt",
        "max_length": 2000,
        "max_input_length": 8000,
    },

    "embed_graph": {
        "enabled": False,
    },

    # Keep rare-but-important entities (e.g., PDAC) in small corpora.
    "prune_graph": {
        "min_node_freq": 1,
        "remove_ego_nodes": False,
    },

    "umap": {
        "enabled": False,
    },

    "snapshots": {
        "graphml": False,
        "embeddings": False,
    },

    "local_search": {
        "chat_model_id": "default_chat_model",
        "embedding_model_id": "default_embedding_model",
        "prompt": "prompts/persona_ventures_candidates.txt",
        "max_context_tokens": 8000,
    },

    "global_search": {
        "chat_model_id": "default_chat_model",
        "map_prompt": "prompts/global_search_map_system_prompt.txt",
        "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt",
        "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt",
    },

    "drift_search": {
        "chat_model_id": "default_chat_model",
        "embedding_model_id": "default_embedding_model",
        "prompt": "prompts/drift_search_system_prompt.txt",
        "reduce_prompt": "prompts/drift_reduce_prompt.txt",
    },

    "basic_search": {
        "chat_model_id": "default_chat_model",
        "embedding_model_id": "default_embedding_model",
        "prompt": "prompts/basic_search_system_prompt.txt",
    },
      
   }
    
  settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
  print(f"[config] settings.yaml created at {settings_path}")

print("----- settings.yaml -----")
print(settings_path.read_text())
print("-------------------------")

# -------------------------------
# 4) Download docs via REST + SAS
#    (reads values from env; FIX: use '&' not '&amp;' in list URL)
# -------------------------------
BLOB_ACCOUNT_URL = os.getenv("BLOB_ACCOUNT_URL", "https://kdfoundry.blob.core.windows.net")
BLOB_CONTAINER   = os.getenv("BLOB_CONTAINER", "crhukinput")
BLOB_PREFIX      = os.getenv("BLOB_PREFIX", "DataSource")
SRC_SAS_TOKEN    = os.getenv("SRC_SAS_TOKEN","sp=racwdli&st=2026-03-02T13:26:50Z&se=2026-03-02T21:41:50Z&spr=https&sv=2024-11-04&sr=c&sig=YPBUY%2FsYIVQnJFmFUrPT8G0UylNdcDYYtDTMaYqVir8%3D")  # supply RAW SAS (no HTML escapes)
BLOB_PROMPT_PREFIX = "prompts"



def list_blobs_with_prefix(account_url, container, prefix, sas):
    # Ensure SAS is raw and NOT escaped
    sas = html.unescape(sas).replace("&amp;", "&").lstrip("?")
    
    # Build correct URL with raw '&'
    url = f"{account_url}/{container}?restype=container&comp=list&prefix={prefix}&{sas}"
    
    print("[debug] LIST URL:", url)  # <--- should show NO &amp; anywhere

    r = requests.get(url, timeout=60)
    if not r.ok:
        print(f"[download] List failed {r.status_code}: {r.text[:500]}")
        r.raise_for_status()

    root = ET.fromstring(r.text)
    return [n.text for n in root.findall('.//{*}Name')]

def download_blob(account_url, container, blob_name, sas, dest_path):
    sas = html.unescape(sas).replace("&amp;", "&").lstrip("?")
    url = f"{account_url}/{container}/{blob_name}?{sas}"

    print("[debug] DOWNLOAD URL:", url)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as r:
        if not r.ok:
            print(f"[download] GET failed {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)



def download_docs_to_input():
    if not SRC_SAS_TOKEN:
        print("[download] No SRC_SAS_TOKEN provided; skipping download.")
        return
    names = list_blobs_with_prefix(BLOB_ACCOUNT_URL, BLOB_CONTAINER, BLOB_PREFIX, SRC_SAS_TOKEN)
    if not names:
        print("[download] No blobs found. Check SAS token and prefix.")
        return
    count = 0
    for name in names:
        if Path(name).suffix.lower() not in [".txt", ".csv", ".md"]:
            continue
        target = INPUT_DIR / Path(name).name
        download_blob(BLOB_ACCOUNT_URL, BLOB_CONTAINER, name, SRC_SAS_TOKEN, target)
        count += 1
        print(f"[download] {name} -> {target}")
    print(f"[download] {count} docs downloaded to {INPUT_DIR}")

download_docs_to_input()



PROMPTS_DIR  = PROJECT_ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

def download_prompts():
    if not SRC_SAS_TOKEN:
        print("[download] No SRC_SAS_TOKEN provided; skipping download.")
        return
    
    names = list_blobs_with_prefix(BLOB_ACCOUNT_URL, BLOB_CONTAINER, BLOB_PROMPT_PREFIX, SRC_SAS_TOKEN)
    if not names:
        print("[download] No Prompts found. Check SAS token and prefix.")
        return
    count = 0
    for name in names:
        if not name.lower().endswith(".txt"):
            continue
        target = PROMPTS_DIR / Path(name).name
        download_blob(BLOB_ACCOUNT_URL, BLOB_CONTAINER, name, SRC_SAS_TOKEN, target)
        count += 1
        print(f"[download] {name} -> {target}")
    print(f"[download] {count} Prompts downloaded to {PROMPTS_DIR}")

download_prompts()


# -------------------------------
# 5) Run GraphRAG indexing
# -------------------------------
def run_graphrag_index(root: Path) -> int:
    print("[index] Running GraphRAG index...")
    config_name = "settings.yaml"

    for folder in [root / "output", root / "cache"]:
        if folder.exists():
            shutil.rmtree(folder)
    cmd = ["python", "-m", "graphrag", "index", "--root", str(root), "--config", config_name]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env)

    print("STDOUT:\n", proc.stdout)
    print("STDERR:\n", proc.stderr)
    print(f"Return code: {proc.returncode}")

    if proc.returncode == 0:
        print("[index] Completed successfully.")
    else:
        print("[index] GraphRAG indexing failed.")

    return proc.returncode

rc = run_graphrag_index(PROJECT_ROOT)

# -------------------------------
# 6) Upload results to Blob Storage (outputs/)
# -------------------------------
DST_ACCOUNT_URL = os.getenv("DST_ACCOUNT_URL", "https://kdfoundry.blob.core.windows.net")
DST_CONTAINER   = os.getenv("DST_CONTAINER", "crhukoutput")
DST_PREFIX      = os.getenv("DST_PREFIX", "graphrag_project5")  # folder path inside container
DST_SAS_TOKEN   = os.getenv("DST_SAS_TOKEN", "sp=racwdli&st=2026-03-02T12:14:13Z&se=2026-03-31T19:29:13Z&spr=https&sv=2024-11-04&sr=c&sig=EPJ9XIwhC1TYIyU%2BADMrUuCv3YmqqSQ6RcKrSwRfqQg%3D")  # RAW SAS (no HTML entities). If not provided, will try Managed Identity.

def upload_dir(path: Path, account_url: str, container: str, prefix: str, sas: str | None = None):
    if not (account_url and container and prefix):
        print("[upload] Skipping — destination settings incomplete.")
        return

    # Lazily import Azure SDK to avoid import cost/requirement if not uploading
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception as e:
        print(f"[upload] azure-storage-blob not installed: {e}")
        return

    # Prepare credential: SAS (preferred) or Managed Identity
    credential = None
    if sas:
        credential = html.unescape(sas).lstrip("?")
        print("[upload] Using SAS credential.")
    else:
        try:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            print("[upload] Using DefaultAzureCredential (Managed Identity / Azure login).")
        except Exception as e:
            print(f"[upload] azure-identity not available; cannot authenticate without SAS. {e}")
            return

    # Create clients
    bsc = BlobServiceClient(account_url=account_url, credential=credential)
    cc = bsc.get_container_client(container)

    # Try to create container if it doesn't exist (requires 'c' permission on SAS)
    try:
        cc.create_container()
        print(f"[upload] Container '{container}' created (or already exists).")
    except Exception:
        # Likely already exists or SAS lacks 'c' permission; continue
        pass

    # Upload files recursively
    file_count = 0
    for f in path.rglob("*"):
        if f.is_file():
            blob_path = f"{prefix}/{f.relative_to(path).as_posix()}"
            with f.open("rb") as fh:
                cc.upload_blob(blob_path, fh, overwrite=True)
            file_count += 1
            print(f"[upload] {f} -> {blob_path}")
    print(f"[upload] Done. {file_count} files uploaded to {account_url}/{container}/{prefix}")

try:
    if OUTPUT_DIR.exists():
        upload_dir(PROJECT_ROOT, DST_ACCOUNT_URL, DST_CONTAINER, DST_PREFIX, DST_SAS_TOKEN)
    else:
        print(f"[upload] Skipped — output directory not found: {OUTPUT_DIR}")
except Exception as e:
    print(f"[upload] Warning: {e}")

# -------------------------------
# 7) LiteLLM cleanup for notebooks/async clients
# -------------------------------
os.environ.setdefault("LITELLM_LOGGING_WORKER_DISABLED", "true")
os.environ.setdefault("LITELLM_LOGGING_PROCESS", "sync")

def run_coro_in_new_loop(coro_func, label: str):
    if not callable(coro_func):
        print(f"[cleanup] {label} not callable; skipped")
        return
    exc_holder = {"exc": None}
    def _runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            coro = coro_func()
            if not inspect.iscoroutine(coro):
                print(f"[cleanup] {label} is not a coroutine; skipped")
                return
            loop.run_until_complete(coro)
        except Exception as e:
            exc_holder["exc"] = e
            traceback.print_exc()
        finally:
            try:
                loop.close()
            except Exception:
                pass
    t = threading.Thread(target=_runner, name=f"cleanup-{label}", daemon=True)
    t.start()
    t.join()
    if exc_holder["exc"] is None:
        print(f"[cleanup] awaited {label}")
    else:
        print(f"[cleanup] {label} skipped due to error: {exc_holder['exc']}")

try:
    from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
    run_coro_in_new_loop(lambda: close_litellm_async_clients(), "litellm async clients")
except Exception as e:
    print(f"[cleanup] litellm async clients import skipped: {e}")

try:
    from litellm.litellm_core_utils.logging_worker import Logging
    async_flush = getattr(Logging, "async_flush", None)
    if async_flush and inspect.iscoroutinefunction(async_flush):
        run_coro_in_new_loop(lambda: async_flush(), "litellm logging flush")
    else:
        async_success_handler = getattr(Logging, "async_success_handler", None)
        if async_success_handler and inspect.iscoroutinefunction(async_success_handler):
            run_coro_in_new_loop(lambda: async_success_handler({}), "litellm logging handler")
        else:
            print("[cleanup] litellm logging has no async flush/handler; skipped")
except Exception as e:
    print(f"[cleanup] litellm logging cleanup skipped: {e}")

print("=== GraphRAG Index Job finished ===")
