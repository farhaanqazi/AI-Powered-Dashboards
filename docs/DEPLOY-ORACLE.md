# Deploying for free on an Oracle "Always Free" cloud computer

This runs the **full four-piece setup** (web app + background worker + Redis +
database) using the `docker-compose.yml` already in this repo — for **$0/month**,
always on, with a public web address.

Hugging Face can't do this: it gives you one container and can't keep a Redis or
a database alive next to it. Oracle's free tier gives you a real Linux machine
where all four pieces run together.

> **The one string attached:** Oracle asks for a credit/debit card at sign-up to
> prove you're a real person. It is **not charged** as long as you stay on the
> "Always Free" machine described below. Don't add paid resources and you pay
> nothing.

---

## 1. Create the free machine

1. Sign up at <https://www.oracle.com/cloud/free/>. Choose a home region close to
   you (you can't change it later).
2. In the console: **Menu → Compute → Instances → Create Instance**.
3. Settings:
   - **Image:** Canonical **Ubuntu 22.04**.
   - **Shape:** click *Change shape* → **Ampere (ARM)** → `VM.Standard.A1.Flex`.
     Set **2 OCPUs** and **12 GB memory** (well within the always-free limit of 4
     OCPU / 24 GB, and plenty for this app).
   - **Networking:** keep "Assign a public IPv4 address" checked.
   - **SSH keys:** let it generate a key pair and **download the private key**.
     You need it to log in.
4. Click **Create**. After a minute you'll see a **public IP address** — note it.

> If Ampere capacity is "out of host capacity" in your region, retry over a few
> hours, or use the smaller always-free AMD shape `VM.Standard.E2.1.Micro`
> (1 CPU / 1 GB — works but slow; ARM is strongly preferred).

---

## 2. Open the web port (two firewalls — both matter)

Oracle blocks traffic in **two** places. You must open port **7860** in both.

**a) Cloud firewall (Security List):**
Console → your instance → click its **Subnet** → click the **Default Security
List** → **Add Ingress Rules**:
- Source CIDR: `0.0.0.0/0`
- IP Protocol: **TCP**
- Destination Port Range: `7860`
- Save.

**b) Host firewall (inside Ubuntu):** Oracle's Ubuntu image ships with a
restrictive built-in firewall. After you log in (next step), run:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 7860 -j ACCEPT
sudo netfilter-persistent save
```

---

## 3. Log in and install Docker

From your laptop (PowerShell), connect with the key you downloaded:

```powershell
ssh -i C:\path\to\your-key.key ubuntu@YOUR_PUBLIC_IP
```

Then, on the server, install Docker + the compose plugin:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker   # apply group change without logging out
```

Confirm it works: `docker run hello-world`.

---

## 4. Get the app onto the server

```bash
git clone YOUR_GITHUB_REPO_URL app
cd app
cp .env.example .env
nano .env
```

Edit `.env` and set at least:

| Setting | What to put |
|---|---|
| `CLERK_PUBLISHABLE_KEY` | Your Clerk publishable key (also used at build time). |
| `GROQ_API_KEY` | Your Groq key, if you use the AI analyst feature. |
| `CORS_ALLOW_ORIGINS` | `http://YOUR_PUBLIC_IP:7860` (add your domain later if you get one). |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Only if you load Kaggle datasets. |

> Leave `JOB_QUEUE_ENABLED`, `REDIS_URL`, `DATABASE_URL` **as the compose file
> sets them** — `docker-compose.yml` already wires the worker, Redis, and
> Postgres correctly. The values in `.env` for those are ignored in favor of the
> compose environment.

---

## 5. Start all four pieces

The frontend needs your Clerk key baked in **at build time**, so pass it as a
build argument:

```bash
export VITE_CLERK_PUBLISHABLE_KEY="pk_live_xxx_your_clerk_key"
docker compose up -d --build
```

That's it. Check it's healthy:

```bash
docker compose ps
docker compose logs -f api    # Ctrl-C to stop watching
```

Open **`http://YOUR_PUBLIC_IP:7860`** in a browser.

---

## 6. Updating later (one command)

When you push new code to GitHub, update the server with:

```bash
cd ~/app
git pull
docker compose up -d --build
```

The database and uploaded-file storage survive restarts (they live in Docker
volumes `pgdata` and `spool` defined in the compose file).

---

## 7. Optional: a real web address + HTTPS (still free)

Right now you're on `http://IP:7860` (no padlock). To get a free domain and
automatic HTTPS:

1. Get a free subdomain at <https://www.duckdns.org> (e.g. `mydash.duckdns.org`)
   and point it at your public IP.
2. Open ports **80** and **443** in both firewalls (same as step 2).
3. Add a small **Caddy** reverse proxy — it gets a free certificate
   automatically. Create `Caddyfile` next to the compose file:

   ```
   mydash.duckdns.org {
       reverse_proxy api:7860
   }
   ```

   …and add a `caddy` service to `docker-compose.yml` on ports 80/443. Ask me and
   I'll write that part for you.

---

## Troubleshooting

- **Page won't load:** 90% of the time it's the firewall. Re-check **both**
  layers in step 2. Test from the server itself: `curl localhost:7860` — if that
  works but the public IP doesn't, it's a firewall.
- **Build runs out of memory:** bump the instance to 4 OCPU / 24 GB (still free),
  or build with `docker compose build --no-cache api` first, then `up -d`.
- **Login (Clerk) broken:** the publishable key must be set **both** in `.env`
  and exported as `VITE_CLERK_PUBLISHABLE_KEY` before `--build` (step 5). Vite
  bakes it into the frontend at build time, so a missing key = broken login.
