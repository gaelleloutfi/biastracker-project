# Deploying BiasTracker as a private website

BiasTracker is a Streamlit app ([`biastracker/app.py`](biastracker/app.py)). Running
`streamlit run app.py` locally serves it on `localhost`; to get a shareable URL you
host it somewhere. This guide covers two **free** options, chosen so that only your
small team of collaborators can use it.

Both options rely on a key property of this app: **uploaded data is processed in
memory via `tempfile` and never written to a database**, so there is no data at rest
to leak. The only security questions are *who can open the app* and *whose servers do
the processing run on*.

| | Data processed on | Always online? | Setup effort |
|---|---|---|---|
| **A. Streamlit Community Cloud (private)** | Streamlit/Snowflake servers (transiently) | ✅ yes | Low |
| **B. Self-host + Cloudflare Tunnel** | **Only your machine** | Only while your machine is on | Medium |

If your data is under a policy that forbids *any* third-party processing, use **B**.
Otherwise **A** is the easier, always-on choice for a small team.

---

## Prerequisites (both options)

1. Push this repo to GitHub (a **private** repo is fine and recommended).
2. The root [`requirements.txt`](requirements.txt) installs both local packages
   (`protperties` is not on PyPI, so it is installed from its path).
3. App entry point / main file: **`biastracker/app.py`**.
4. Python **3.11 or 3.12** (biastracker requires >= 3.10).

---

## Option A — Streamlit Community Cloud (private, free, always-on)

1. Go to <https://share.streamlit.io> and sign in with the **GitHub account** that owns
   the repo. Authorize access to the (private) repo.
2. **New app** → pick this repo and branch (`main`).
   - **Main file path:** `biastracker/app.py`
   - **Advanced settings → Python version:** 3.12
3. Click **Deploy**. First build takes a few minutes (it installs both local packages).
4. **Lock it down:** open the app's **⋮ → Settings → Sharing** and set
   **"Only specific people can view this app"**, then add each collaborator's email.
   They sign in with Google/GitHub; anyone not on the list cannot open the app.

**Security summary:** access is restricted to named emails; uploads are transient and
never persisted. The only exposure is that files are processed on Streamlit's cloud
during a session — fine for typical research data, not for data barred from third-party
clouds (use Option B for that).

---

## Option B — Self-host + Cloudflare Tunnel + Access (data never leaves your machine)

Run the app on a machine you control (ideally an always-on lab workstation/server, not a
laptop that sleeps) and expose it through an encrypted Cloudflare Tunnel gated by a login.

**One-time setup on the host machine:**

```bash
# 1. Install the packages (from the repo root)
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run the app headless on localhost
streamlit run biastracker/app.py --server.port 8501 --server.headless true
```

**Expose it securely (free Cloudflare account + a domain on Cloudflare):**

1. Install `cloudflared` and authenticate: `cloudflared tunnel login`.
2. Create a named tunnel and route a hostname (e.g. `biastracker.yourdomain.org`) to
   `http://localhost:8501`:
   ```bash
   cloudflared tunnel create biastracker
   cloudflared tunnel route dns biastracker biastracker.yourdomain.org
   cloudflared tunnel run biastracker
   ```
3. In the **Cloudflare Zero Trust dashboard → Access → Applications**, add a
   self-hosted app for that hostname with a policy that **allows only your
   collaborators' email addresses**. Cloudflare handles the login screen (one-time PIN
   or SSO); the tunnel is end-to-end encrypted and your machine's port is never exposed
   directly.

> A quick `cloudflared tunnel --url http://localhost:8501` gives an instant
> `*.trycloudflare.com` URL but has **no authentication** — use it only for a throwaway
> demo, never for real data.

**Security summary:** uploaded data is processed only on your hardware; the tunnel is
encrypted and access is restricted to named emails. Trade-off: the site is only up while
your host machine and `cloudflared` are running.

---

## Optional: further slim the repo before deploying

The app needs these bundled reference files (they must stay):
`biastracker/data/raw/hpa/subcellular_location_long_uniprot.csv` (~14 MB),
`.../contaminants/contaminants_long.csv`, `.../paxdb/human_abundance_uniprot.csv`,
and `biastracker/assets/logo.png`.

It does **not** need the large `DVPT191203` sample datasets under
`biastracker/data/raw/maxquant/` and `biastracker/data/processed/` (~130 MB of dev/demo
data users don't need, since they upload their own). Removing them speeds up every clone
and keeps you well under host limits. Do this in a follow-up if you want a leaner deploy.
