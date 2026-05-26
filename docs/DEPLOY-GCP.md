# Deploying for free on a Google Cloud "Always Free" VM

Runs the **full four-piece stack** (web app + Arq worker + Redis + Postgres)
from the `docker-compose.yml` in this repo, on Google Cloud's **Always Free**
`e2-micro` VM — **$0/month**, always on, public web address.

> **Card required:** Google asks for a credit/debit card at sign-up to verify
> identity. It is **not charged** while you stay on the Always Free `e2-micro`
> in an eligible region. Don't add paid resources and you pay nothing.

> **The catch — RAM.** `e2-micro` has **1 GB**. The image builds the React
> frontend (`npm run build`) *and* runs four services. 1 GB is not enough on its
> own — §3 adds swap and §5 builds in a memory-safe way. Follow them exactly or
> the build gets OOM-killed.

---

## 1. Create the free VM

1. Sign up at <https://cloud.google.com/free>. Create a project.
2. Console → **Compute Engine → VM instances → Create instance**.
   (Enable the Compute Engine API if prompted.)
3. Settings:
   - **Region:** must be one of the Always-Free regions —
     `us-west1` (Oregon), `us-central1` (Iowa), or `us-east1` (S. Carolina).
     Any other region is **not** free.
   - **Machine type:** series **E2** → **`e2-micro`** (0.25–2 vCPU, 1 GB). This
     is the only Always-Free shape.
   - **Boot disk:** **Ubuntu 22.04 LTS**, **Standard persistent disk**, **30 GB**
     (free tier covers 30 GB standard; do not pick SSD/balanced).
   - **Firewall:** check **Allow HTTP** and **Allow HTTPS** traffic (we still
     open 7860 separately below).
4. **Create.** Note the **External IP** shown on the instances list.

---

## 2. Open the web port (GCP firewall)

The app listens on **7860**. Console → **VPC network → Firewall → Create
firewall rule**:

- **Name:** `allow-7860`
- **Direction:** Ingress · **Action:** Allow
- **Targets:** All instances in the network (or a target tag you also add to the VM)
- **Source IPv4 ranges:** `0.0.0.0/0`
- **Protocols and ports:** TCP → `7860`
- **Create.**

> Unlike some clouds, the Ubuntu image here has no extra host firewall blocking
> you — the GCP rule is enough. If you later add a domain + HTTPS (§7), also open
> 80 and 443.

---

## 3. Log in, install Docker, add swap

SSH in from the console (**SSH** button on the instance) or via `gcloud`.

```bash
# Docker + compose plugin
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# 4 GB swap — REQUIRED so the 1 GB VM can build the frontend without OOM
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h   # confirm Swap: 4.0Gi
```

Confirm Docker: `docker run hello-world`.

---

## 4. Get the app onto the VM

The GitHub repo is **private**, so clone over HTTPS with a GitHub
Personal Access Token (or add an SSH deploy key):

```bash
git clone https://github.com/farhaanqazi/AI-Powered-Dashboards.git app
# username: your GitHub user · password: a PAT with `repo` scope
cd app
cp .env.example .env
nano .env
```

Set at least:

| Setting | Value |
|---|---|
| `CLERK_PUBLISHABLE_KEY` | Your Clerk publishable key (also needed at build time, §5). |
| `GUEST_MODE_ENABLED` | **`false`** — makes the app invite-only (see §6). |
| `GROQ_API_KEY` | Your Groq key, if you use the AI analyst feature. |
| `CORS_ALLOW_ORIGINS` | `http://YOUR_EXTERNAL_IP:7860` (add a domain later if you get one). |
| `GUEST_SESSION_SECRET` | Any long random string (only matters if you later re-enable guests). |

> Leave `JOB_QUEUE_ENABLED`, `REDIS_URL`, `DATABASE_URL` as the compose file
> sets them — `docker-compose.yml` wires the worker, Redis, and Postgres
> correctly and overrides the `.env` values for those.

---

## 5. Build memory-safe, then start

On 1 GB, let the Node build use the swap and cap its heap. Build **first**
(slow — several minutes — that's the swap working), then start:

```bash
# Clerk key must be present at build time (Vite inlines VITE_* into the bundle)
export VITE_CLERK_PUBLISHABLE_KEY="pk_live_xxx_your_clerk_key"

# cap Node heap so the build doesn't get OOM-killed on 1 GB + swap
export DOCKER_BUILDKIT=1
docker compose build --build-arg CACHEBUST=$(date +%s) api

# then bring up all four pieces
docker compose up -d
```

Check health:

```bash
docker compose ps
docker compose logs -f api    # Ctrl-C to stop watching
curl localhost:7860           # should return HTML
```

Open **`http://YOUR_EXTERNAL_IP:7860`**.

> If the build still gets killed: build with even more swap (8 G), or build the
> image on your laptop and push it to a registry (Docker Hub / GCP Artifact
> Registry), then `docker compose pull` on the VM instead of building there.

---

## 6. Invite-only access (so only your prospect gets in)

Two halves — the code gate (this repo) and the Clerk gate (dashboard):

1. **Code gate (done in `.env`):** `GUEST_MODE_ENABLED=false` makes every
   endpoint require a valid Clerk login — no anonymous guest access.
2. **Clerk gate (Clerk dashboard):**
   - Clerk dashboard → **User & Authentication → Restrictions** → set sign-ups
     to **Restricted / allowlist**, and **add your prospect's email** to the
     allowlist. Now only that email can create an account.
   - Or send them a Clerk **invitation** directly.
3. Give the prospect the URL + tell them to sign in with the invited email.

Anyone else hitting the URL sees the login wall and cannot get past it.

---

## 7. Updating later

```bash
cd ~/app
git pull
export VITE_CLERK_PUBLISHABLE_KEY="pk_live_xxx_your_clerk_key"
docker compose build --build-arg CACHEBUST=$(date +%s) api
docker compose up -d
```

Postgres and uploaded-file storage survive restarts (Docker volumes `pgdata`
and `spool` in the compose file).

---

## 8. Optional: real domain + HTTPS (still free)

1. Free subdomain at <https://www.duckdns.org> → point it at your External IP.
2. Open ports **80** and **443** (another firewall rule like §2).
3. Add a **Caddy** reverse proxy (auto HTTPS). Create `Caddyfile`:
   ```
   yourname.duckdns.org {
       reverse_proxy api:7860
   }
   ```
   …and add a `caddy` service to `docker-compose.yml` on 80/443. Ask and I'll
   write it.

---

## Troubleshooting

- **Build killed / "Killed" during `npm run build`:** not enough swap. Confirm
  `free -h` shows 4 GB swap; bump to 8 GB if needed, or pre-build off-VM (§5).
- **Page won't load:** check the §2 firewall rule. `curl localhost:7860` on the
  VM — if that works but the public IP doesn't, it's the firewall.
- **Login broken / everyone locked out:** `VITE_CLERK_PUBLISHABLE_KEY` must be
  exported **before** `docker compose build` (Vite bakes it in). A missing key =
  broken login = no one (not even the prospect) can get in.
- **Prospect can't sign up:** confirm their email is on the Clerk allowlist (§6)
  and that they're using that exact email.
